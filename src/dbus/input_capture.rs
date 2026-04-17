use calloop::channel;
use std::collections::HashMap;
use std::os::fd::FromRawFd;
use std::os::unix::io::IntoRawFd;
use std::sync::{Arc, Mutex};

/// Barrier definition: axis-aligned line segment on output edge
#[derive(Debug, Clone)]
pub struct Barrier {
    pub id: u32,
    pub x1: i32,
    pub y1: i32,
    pub x2: i32,
    pub y2: i32,
}

/// Zone representing an output area
#[derive(Debug, Clone)]
pub struct Zone {
    pub width: u32,
    pub height: u32,
    pub x: i32,
    pub y: i32,
}

/// State of an input capture session
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CaptureSessionState {
    Created,
    Disabled,
    Enabled,
    Activated,
}

/// EIS connection state for a session (server-side).
///
/// All reis types (Connection, Seat, Device) use Arc internally and are Clone.
/// They may not be Send/Sync in all configurations, so we wrap in a
/// non-Send marker if needed. In practice, since the compositor is
/// single-threaded on the calloop event loop, and we only access these
/// from the event loop callbacks, this is safe.
pub struct EisConnection {
    pub connection: reis::request::Connection,
    pub seat: reis::request::Seat,
    pub pointer_device: reis::request::Device,
    pub keyboard_device: reis::request::Device,
    /// Sequence counter for start_emulating
    pub sequence: u32,
}

impl std::fmt::Debug for EisConnection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EisConnection")
            .field("sequence", &self.sequence)
            .finish_non_exhaustive()
    }
}

// SAFETY: EisConnection is only accessed from the compositor's calloop event loop thread.
// The reis types use Arc<Mutex<...>> internally for their state, making them
// structurally safe to send between threads, even though they don't explicitly
// implement Send. We need Send+Sync because they're stored in InputCaptureState
// which is behind Arc<Mutex<...>>.
unsafe impl Send for EisConnection {}
unsafe impl Sync for EisConnection {}

/// Per-session capture data
#[derive(Debug)]
pub struct CaptureSession {
    pub state: CaptureSessionState,
    pub capabilities: u32,
    pub barriers: Vec<Barrier>,
    pub zone_set: u32,
    pub activation_id: u32,
    pub eis_fd: Option<std::os::fd::OwnedFd>,
    pub eis_connection: Option<EisConnection>,
}

/// D-Bus signal events sent from compositor to the D-Bus service thread
#[derive(Debug)]
pub enum SignalEvent {
    Activated {
        session_id: String,
        barrier_id: u32,
        activation_id: u32,
        cursor_position: (f64, f64),
    },
    Deactivated {
        session_id: String,
        activation_id: u32,
    },
    Disabled {
        session_id: String,
    },
    ZonesChanged {
        zone_set: u32,
    },
}

/// Messages from the D-Bus interface to the compositor event loop
#[derive(Debug)]
pub enum InputCaptureEvent {
    SessionCreated {
        session_id: String,
        capabilities: u32,
    },
    SessionClosed {
        session_id: String,
    },
    BarriersSet {
        session_id: String,
        barriers: Vec<Barrier>,
        zone_set: u32,
    },
    Enabled {
        session_id: String,
    },
    Disabled {
        session_id: String,
    },
    Released {
        session_id: String,
        activation_id: u32,
        cursor_position: Option<(f64, f64)>,
    },
    ConnectEIS {
        session_id: String,
        server_fd: std::os::fd::OwnedFd,
    },
}

/// Shared state between D-Bus thread and compositor
pub struct InputCaptureState {
    pub sessions: HashMap<String, CaptureSession>,
    /// Currently active capture session (if any)
    pub active_session: Option<String>,
    /// Monotonically increasing zone_set counter
    pub zone_set: u32,
    /// Current output zones
    pub zones: Vec<Zone>,
    /// Next activation ID
    pub next_activation_id: u32,
    /// Channel sender for D-Bus signal emission (compositor -> D-Bus thread)
    pub signal_tx: Option<std::sync::mpsc::Sender<SignalEvent>>,
}

impl std::fmt::Debug for InputCaptureState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InputCaptureState")
            .field("sessions", &self.sessions)
            .field("active_session", &self.active_session)
            .field("zone_set", &self.zone_set)
            .field("zones", &self.zones)
            .field("next_activation_id", &self.next_activation_id)
            .field("signal_tx", &self.signal_tx.is_some())
            .finish()
    }
}

impl Default for InputCaptureState {
    fn default() -> Self {
        Self {
            sessions: HashMap::new(),
            active_session: None,
            zone_set: 0,
            zones: Vec::new(),
            next_activation_id: 0,
            signal_tx: None,
        }
    }
}

impl InputCaptureState {
    pub fn new() -> Self {
        Self {
            zone_set: 1,
            ..Default::default()
        }
    }

    /// Update zones from output layout
    pub fn update_zones(&mut self, zones: Vec<Zone>) {
        self.zones = zones;
        self.zone_set += 1;
    }

    /// Send a signal event to the D-Bus thread for emission
    pub fn emit_signal(&self, event: SignalEvent) {
        if let Some(ref tx) = self.signal_tx {
            if let Err(e) = tx.send(event) {
                tracing::warn!("Failed to send signal event to D-Bus thread: {}", e);
            }
        } else {
            tracing::warn!("No signal_tx configured, cannot emit D-Bus signal");
        }
    }

    /// Check if cursor movement from `from` to `to` crosses any barrier in enabled sessions
    /// Returns (barrier_id, session_id, intersection_point) if a crossing is detected
    pub fn check_barrier_crossing(
        &self,
        from: (f64, f64),
        to: (f64, f64),
    ) -> Option<(u32, String, (f64, f64))> {
        for (session_id, session) in &self.sessions {
            if session.state != CaptureSessionState::Enabled {
                continue;
            }
            for barrier in &session.barriers {
                if let Some(intersection) = line_segment_intersection(
                    from,
                    to,
                    (barrier.x1 as f64, barrier.y1 as f64),
                    (barrier.x2 as f64, barrier.y2 as f64),
                ) {
                    return Some((barrier.id, session_id.clone(), intersection));
                }
            }
        }
        None
    }
}

/// 2D line segment intersection test
/// Returns the intersection point if segments (p1->p2) and (p3->p4) intersect
fn line_segment_intersection(
    p1: (f64, f64),
    p2: (f64, f64),
    p3: (f64, f64),
    p4: (f64, f64),
) -> Option<(f64, f64)> {
    let d1x = p2.0 - p1.0;
    let d1y = p2.1 - p1.1;
    let d2x = p4.0 - p3.0;
    let d2y = p4.1 - p3.1;

    let denom = d1x * d2y - d1y * d2x;

    // Parallel lines
    if denom.abs() < 1e-10 {
        return None;
    }

    let t = ((p3.0 - p1.0) * d2y - (p3.1 - p1.1) * d2x) / denom;
    let u = ((p3.0 - p1.0) * d1y - (p3.1 - p1.1) * d1x) / denom;

    // Check if intersection is within both segments
    if (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u) {
        let x = p1.0 + t * d1x;
        let y = p1.1 + t * d1y;
        Some((x, y))
    } else {
        None
    }
}

/// D-Bus interface served by the compositor
pub struct InputCaptureInterface {
    state: Arc<Mutex<InputCaptureState>>,
    tx: channel::Sender<InputCaptureEvent>,
}

impl InputCaptureInterface {
    pub fn new(
        state: Arc<Mutex<InputCaptureState>>,
        tx: channel::Sender<InputCaptureEvent>,
    ) -> Self {
        Self { state, tx }
    }
}

#[zbus::interface(name = "org.cosmic.InputCapture")]
impl InputCaptureInterface {
    async fn create_session(&self, capabilities: u32) -> zbus::fdo::Result<String> {
        let session_id = format!("input_capture_{}", uuid_simple());

        {
            let mut state = self.state.lock().unwrap();
            let current_zone_set = state.zone_set;
            state.sessions.insert(
                session_id.clone(),
                CaptureSession {
                    state: CaptureSessionState::Created,
                    capabilities,
                    barriers: Vec::new(),
                    zone_set: current_zone_set,
                    activation_id: 0,
                    eis_fd: None,
                    eis_connection: None,
                },
            );
        }

        let _ = self.tx.send(InputCaptureEvent::SessionCreated {
            session_id: session_id.clone(),
            capabilities,
        });

        Ok(session_id)
    }

    async fn get_zones(
        &self,
        _session_id: &str,
    ) -> zbus::fdo::Result<(u32, Vec<(u32, u32, i32, i32)>)> {
        let state = self.state.lock().unwrap();
        let zones: Vec<(u32, u32, i32, i32)> = state
            .zones
            .iter()
            .map(|z| (z.width, z.height, z.x, z.y))
            .collect();
        Ok((state.zone_set, zones))
    }

    async fn set_barriers(
        &self,
        session_id: &str,
        zone_set: u32,
        barriers: Vec<(u32, (i32, i32, i32, i32))>,
    ) -> zbus::fdo::Result<Vec<u32>> {
        let mut state = self.state.lock().unwrap();

        if state.zone_set != zone_set {
            return Err(zbus::fdo::Error::Failed("Zone set mismatch".into()));
        }

        let session = state
            .sessions
            .get_mut(session_id)
            .ok_or_else(|| zbus::fdo::Error::Failed("Session not found".into()))?;

        let mut valid_barriers = Vec::new();
        let mut failed = Vec::new();

        for (id, (x1, y1, x2, y2)) in &barriers {
            // Must be axis-aligned
            if x1 != x2 && y1 != y2 {
                failed.push(*id);
                continue;
            }
            // Must not be a point
            if x1 == x2 && y1 == y2 {
                failed.push(*id);
                continue;
            }

            valid_barriers.push(Barrier {
                id: *id,
                x1: *x1,
                y1: *y1,
                x2: *x2,
                y2: *y2,
            });
        }

        session.barriers = valid_barriers.clone();
        session.zone_set = zone_set;

        drop(state);

        let _ = self.tx.send(InputCaptureEvent::BarriersSet {
            session_id: session_id.to_string(),
            barriers: valid_barriers,
            zone_set,
        });

        Ok(failed)
    }

    async fn enable(&self, session_id: &str) -> zbus::fdo::Result<()> {
        let mut state = self.state.lock().unwrap();
        let session = state
            .sessions
            .get_mut(session_id)
            .ok_or_else(|| zbus::fdo::Error::Failed("Session not found".into()))?;

        match session.state {
            CaptureSessionState::Created | CaptureSessionState::Disabled => {
                session.state = CaptureSessionState::Enabled;
            }
            _ => {
                return Err(zbus::fdo::Error::Failed(format!(
                    "Invalid state for Enable: {:?}",
                    session.state
                )));
            }
        }

        drop(state);

        let _ = self.tx.send(InputCaptureEvent::Enabled {
            session_id: session_id.to_string(),
        });

        Ok(())
    }

    async fn disable(&self, session_id: &str) -> zbus::fdo::Result<()> {
        let mut state = self.state.lock().unwrap();
        let session = state
            .sessions
            .get_mut(session_id)
            .ok_or_else(|| zbus::fdo::Error::Failed("Session not found".into()))?;

        session.state = CaptureSessionState::Disabled;

        drop(state);

        let _ = self.tx.send(InputCaptureEvent::Disabled {
            session_id: session_id.to_string(),
        });

        Ok(())
    }

    async fn connect_to_eis(&self, session_id: &str) -> zbus::fdo::Result<zbus::zvariant::OwnedFd> {
        let (client_stream, server_stream) = std::os::unix::net::UnixStream::pair()
            .map_err(|e| zbus::fdo::Error::Failed(format!("socketpair failed: {}", e)))?;

        let server_fd = unsafe { std::os::fd::OwnedFd::from_raw_fd(server_stream.into_raw_fd()) };
        let client_fd = unsafe { std::os::fd::OwnedFd::from_raw_fd(client_stream.into_raw_fd()) };

        let _ = self.tx.send(InputCaptureEvent::ConnectEIS {
            session_id: session_id.to_string(),
            server_fd,
        });

        Ok(zbus::zvariant::OwnedFd::from(client_fd))
    }

    async fn release(
        &self,
        session_id: &str,
        activation_id: u32,
        cursor_position: (f64, f64),
    ) -> zbus::fdo::Result<()> {
        let mut state = self.state.lock().unwrap();
        let session = state
            .sessions
            .get_mut(session_id)
            .ok_or_else(|| zbus::fdo::Error::Failed("Session not found".into()))?;

        session.state = CaptureSessionState::Disabled;
        if let Some(ref active) = state.active_session {
            if active == session_id {
                state.active_session = None;
            }
        }

        drop(state);

        let _ = self.tx.send(InputCaptureEvent::Released {
            session_id: session_id.to_string(),
            activation_id,
            cursor_position: Some(cursor_position),
        });

        Ok(())
    }

    async fn close(&self, session_id: &str) -> zbus::fdo::Result<()> {
        let mut state = self.state.lock().unwrap();
        state.sessions.remove(session_id);
        if state.active_session.as_deref() == Some(session_id) {
            state.active_session = None;
        }
        drop(state);

        let _ = self.tx.send(InputCaptureEvent::SessionClosed {
            session_id: session_id.to_string(),
        });

        Ok(())
    }

    // Signals
    #[zbus(signal)]
    async fn activated(
        signal_ctxt: &zbus::object_server::SignalEmitter<'_>,
        session_id: &str,
        barrier_id: u32,
        activation_id: u32,
        cursor_position: (f64, f64),
    ) -> zbus::Result<()>;

    #[zbus(signal)]
    async fn deactivated(
        signal_ctxt: &zbus::object_server::SignalEmitter<'_>,
        session_id: &str,
        activation_id: u32,
    ) -> zbus::Result<()>;

    #[zbus(signal)]
    async fn disabled_signal(
        signal_ctxt: &zbus::object_server::SignalEmitter<'_>,
        session_id: &str,
    ) -> zbus::Result<()>;

    #[zbus(signal)]
    async fn zones_changed(
        signal_ctxt: &zbus::object_server::SignalEmitter<'_>,
        zone_set: u32,
    ) -> zbus::Result<()>;
}

/// Generate a simple unique ID
fn uuid_simple() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{:x}", nanos)
}

/// Initialize the InputCapture D-Bus service and return an event source + shared state
pub fn init(
    executor: &futures_executor::ThreadPool,
) -> anyhow::Result<(
    channel::Channel<InputCaptureEvent>,
    Arc<Mutex<InputCaptureState>>,
)> {
    let (tx, rx) = channel::channel::<InputCaptureEvent>();
    let state = Arc::new(Mutex::new(InputCaptureState::new()));
    let state_clone = state.clone();

    // Create a channel for compositor -> D-Bus signal emission
    let (signal_tx, signal_rx) = std::sync::mpsc::channel::<SignalEvent>();

    // Store the signal sender in the shared state
    {
        let mut s = state.lock().unwrap();
        s.signal_tx = Some(signal_tx);
    }

    executor.spawn_ok(async move {
        match serve(state_clone, tx, signal_rx).await {
            Ok(()) => {}
            Err(err) => {
                tracing::error!(?err, "InputCapture D-Bus service failed");
            }
        }
    });

    Ok((rx, state))
}

async fn serve(
    state: Arc<Mutex<InputCaptureState>>,
    tx: channel::Sender<InputCaptureEvent>,
    signal_rx: std::sync::mpsc::Receiver<SignalEvent>,
) -> anyhow::Result<()> {
    let connection = zbus::Connection::session().await?;
    let interface = InputCaptureInterface::new(state, tx);

    connection
        .object_server()
        .at("/org/cosmic/InputCapture", interface)
        .await?;

    connection.request_name("org.cosmic.InputCapture").await?;

    tracing::info!("InputCapture D-Bus service started");

    // Spawn a blocking thread that receives signal events from the compositor
    // and emits D-Bus signals. We use std::sync::mpsc::Receiver::recv() which
    // blocks the thread, avoiding busy-waiting. The D-Bus connection is
    // thread-safe and can be used from this background thread.
    let conn = connection.clone();
    std::thread::Builder::new()
        .name("input-capture-signals".into())
        .spawn(move || {
            // Use a blocking runtime to emit signals
            futures_executor::block_on(async move {
                loop {
                    // Block until a signal event arrives
                    let event = match signal_rx.recv() {
                        Ok(event) => event,
                        Err(_) => {
                            tracing::info!("Signal channel disconnected, stopping signal emission");
                            break;
                        }
                    };

                    let iface_ref = conn
                        .object_server()
                        .interface::<_, InputCaptureInterface>("/org/cosmic/InputCapture")
                        .await;
                    let iface_ref = match iface_ref {
                        Ok(r) => r,
                        Err(e) => {
                            tracing::error!(
                                "Failed to get interface ref for signal emission: {}",
                                e
                            );
                            continue;
                        }
                    };
                    let ctxt = iface_ref.signal_emitter();
                    match event {
                        SignalEvent::Activated {
                            session_id,
                            barrier_id,
                            activation_id,
                            cursor_position,
                        } => {
                            if let Err(e) = InputCaptureInterface::activated(
                                &ctxt,
                                &session_id,
                                barrier_id,
                                activation_id,
                                cursor_position,
                            )
                            .await
                            {
                                tracing::warn!("Failed to emit Activated signal: {}", e);
                            } else {
                                tracing::info!(
                                    session_id,
                                    barrier_id,
                                    activation_id,
                                    "Emitted Activated D-Bus signal"
                                );
                            }
                        }
                        SignalEvent::Deactivated {
                            session_id,
                            activation_id,
                        } => {
                            if let Err(e) = InputCaptureInterface::deactivated(
                                &ctxt,
                                &session_id,
                                activation_id,
                            )
                            .await
                            {
                                tracing::warn!("Failed to emit Deactivated signal: {}", e);
                            } else {
                                tracing::info!(
                                    session_id,
                                    activation_id,
                                    "Emitted Deactivated D-Bus signal"
                                );
                            }
                        }
                        SignalEvent::Disabled { session_id } => {
                            if let Err(e) =
                                InputCaptureInterface::disabled_signal(&ctxt, &session_id).await
                            {
                                tracing::warn!("Failed to emit Disabled signal: {}", e);
                            }
                        }
                        SignalEvent::ZonesChanged { zone_set } => {
                            if let Err(e) =
                                InputCaptureInterface::zones_changed(&ctxt, zone_set).await
                            {
                                tracing::warn!("Failed to emit ZonesChanged signal: {}", e);
                            }
                        }
                    }
                }
            });
        })
        .expect("Failed to spawn signal emission thread");

    // Keep the connection alive
    std::future::pending::<()>().await;
    Ok(())
}
