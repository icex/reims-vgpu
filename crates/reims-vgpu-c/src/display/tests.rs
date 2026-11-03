use reims_vgpu_core::guest::records::APV_PAGE_SHIFT;

use super::*;
use crate::status::*;
use crate::testkit::{Rec, Rig, t_destroy};

unsafe extern "C-unwind" fn t_mode_change(ctx: *mut c_void, w: u16, h: u16, format: u32) {
    Rec::push(ctx, format!("mode_change {w}x{h} {format:#x}"));
}
unsafe extern "C-unwind" fn t_new_frame(ctx: *mut c_void) {
    Rec::push(ctx, "new_frame".to_owned());
}
unsafe extern "C-unwind" fn t_glyph(ctx: *mut c_void, g: *const RvgCursorGlyph) {
    // SAFETY: the glyph is live for the call.
    let g = unsafe { &*g };
    // SAFETY: its pixels, as above.
    let px = unsafe { std::slice::from_raw_parts(g.rgba, g.rgba_length) };
    Rec::push(
        ctx,
        format!(
            "glyph {} {}x{} at {},{} {px:?}",
            g.struct_size, g.width, g.height, g.hot_spot.x, g.hot_spot.y
        ),
    );
}
unsafe extern "C-unwind" fn t_show(ctx: *mut c_void, show: u8) {
    Rec::push(ctx, format!("show {show}"));
}
unsafe extern "C-unwind" fn t_move(ctx: *mut c_void) {
    Rec::push(ctx, "move".to_owned());
}

fn table(rec: &Arc<Rec>) -> RvgDisplayCallbacks {
    RvgDisplayCallbacks {
        struct_size: sized::size_u32::<RvgDisplayCallbacks>(),
        ctx: rec.ctx(),
        destroy: Some(t_destroy),
        mode_change: Some(t_mode_change),
        new_frame: Some(t_new_frame),
        cursor_glyph: Some(t_glyph),
        cursor_show: Some(t_show),
        cursor_move: Some(t_move),
    }
}

fn config(port: u64, serial: u32, cb: &RvgDisplayCallbacks) -> RvgDisplayConfig {
    RvgDisplayConfig {
        struct_size: sized::size_u32::<RvgDisplayConfig>(),
        serial_number: serial,
        port,
        name: std::ptr::null(),
        name_length: 0,
        width_millimeters: 0.0,
        height_millimeters: 0.0,
        callbacks: cb,
        window: std::ptr::null(),
    }
}

fn create(device: *mut RvgDevice, c: &RvgDisplayConfig) -> (RvgStatus, *mut RvgDisplay) {
    let mut d = std::ptr::null_mut();
    // SAFETY: a live device or NULL, a live config; `d` is writable.
    let s = unsafe { rvg_display_create(device, c, &mut d) };
    (s, d)
}

/// Each closure calls its C function with the table's ctx and the arguments core passes; the
/// glyph crosses with its struct size, hot spot and pixels
#[test]
fn callbacks_forward() {
    let rec = Rec::new();
    let t = Table::new(rec.ctx(), None);
    let cb = display_callbacks(&t, &table(&rec));
    (cb.mode_change.unwrap())(640, 480, 0x4247_5241);
    (cb.new_frame.unwrap())();
    (cb.cursor_glyph.unwrap())(&CursorGlyph {
        width: 1,
        height: 1,
        hot_spot: (3, 4),
        rgba: &[1, 2, 3, 4],
    });
    (cb.cursor_show.unwrap())(2);
    (cb.cursor_move.unwrap())();
    assert_eq!(
        rec.lines(),
        vec![
            "mode_change 640x480 0x42475241".to_owned(),
            "new_frame".to_owned(),
            format!(
                "glyph {} 1x1 at 3,4 [1, 2, 3, 4]",
                size_of::<RvgCursorGlyph>()
            ),
            "show 2".to_owned(),
            "move".to_owned(),
        ]
    );
    let none = RvgDisplayCallbacks {
        mode_change: None,
        new_frame: None,
        cursor_glyph: None,
        cursor_show: None,
        cursor_move: None,
        ..table(&rec)
    };
    let cb = display_callbacks(&t, &none);
    assert!(cb.mode_change.is_none() && cb.new_frame.is_none() && cb.cursor_glyph.is_none());
    assert!(cb.cursor_show.is_none() && cb.cursor_move.is_none());
}

/// The config reaches core: port, serial number, name (lossy UTF-8; NULL is core's default),
/// size; the modes keep their flags; the queries answer core's values
#[test]
fn config_and_queries() {
    let rig = Rig::new(2);
    let rec = Rec::new();
    let cb = table(&rec);
    let mut c = config(1, 0x5eed_c001, &cb);
    let name = b"Caf\xc3\xa9 \xff";
    c.name = name.as_ptr().cast();
    c.name_length = name.len();
    c.width_millimeters = 400.0;
    c.height_millimeters = 300.0;
    let (s, d) = create(rig.device, &c);
    assert_eq!(s, RVG_OK);
    // SAFETY: the display made above.
    let core = unsafe { &*d }.display.core();
    assert_eq!((core.port(), core.serial_num()), (1, 0x5eed_c001));
    assert_eq!(core.name().map(|n| n.as_str()), Some("Café \u{fffd}"));
    assert_eq!(core.size_in_millimeters(), (400.0, 300.0));
    let modes = [
        RvgDisplayMode {
            width: 1280,
            height: 720,
            flags: 0,
            refresh_rate_hz: 60.0,
        },
        RvgDisplayMode {
            width: 1920,
            height: 1080,
            flags: RVG_DISPLAY_MODE_EDR | RVG_DISPLAY_MODE_10BPC | 0x100,
            refresh_rate_hz: 120.0,
        },
    ];
    // SAFETY: the live display; two modes.
    unsafe { rvg_display_set_modes(d, modes.as_ptr(), 2) };
    let got = core.mode_list();
    assert_eq!(got.len(), 2);
    assert!(!got[0].supports_edr && !got[0].supports_10bpc);
    assert!(got[1].supports_edr && got[1].supports_10bpc);
    assert_eq!(got[1].refresh_rate_in_hz, 120.0);
    // SAFETY: as above: no modes changes nothing.
    unsafe { rvg_display_set_modes(d, std::ptr::null(), 0) };
    assert_eq!(core.mode_list().len(), 2);
    // SAFETY: as above.
    let pos = unsafe { rvg_display_cursor_position(d) };
    assert_eq!((pos.x, pos.y), (0xffff, 0xffff));
    let (mut g, mut h) = (9, 9);
    // SAFETY: as above.
    unsafe {
        rvg_display_present_counts(d, &mut g, &mut h);
        rvg_display_present_counts(d, std::ptr::null_mut(), std::ptr::null_mut());
    }
    assert_eq!((g, h), (0, 0));
    assert!(rec.destroys().is_empty());
    // SAFETY: the display's reference.
    unsafe { rvg_display_release(d) };
    let c = config(0, 0x5eed_c002, &cb);
    let (s, d) = create(rig.device, &c);
    assert_eq!(s, RVG_OK);
    // SAFETY: the display made above.
    let name = unsafe { &*d }
        .display
        .core()
        .name()
        .map(|n| n.as_str().to_owned());
    assert_eq!(name.as_deref(), Some("Apple Virtual"));
    // SAFETY: as above.
    unsafe {
        rvg_display_release(d);
        rvg_display_release(std::ptr::null_mut());
    }
    // Each display's table ends once, maybe on the display thread after its release returned.
    assert_eq!(rec.wait_destroys(2).len(), 2);
}

/// Each mode's flags reach the guest's display page for the port: RVG_DISPLAY_MODE_EDR is bit 0
/// and RVG_DISPLAY_MODE_10BPC bit 1 of the flags byte of the mode's entry, and a bit the library
/// does not define is neither
#[test]
fn mode_flags_reach_the_guest_s_display_page() {
    use reims_vgpu_core::display::page as sp;
    let rig = Rig::new(1);
    // The guest's root node, then its display page for port 0 at guest page 0x10.
    // SAFETY: the rig's live device.
    unsafe { crate::device::rvg_device_mmio_write(rig.device, 0x101c, 1) };
    // SAFETY: as above.
    let core = unsafe { &*rig.device }.device.core();
    core.set_shared_display_state_page(0x10, 0, false).unwrap();
    let rec = Rec::new();
    let cb = table(&rec);
    let (s, d) = create(rig.device, &config(0, 0x5eed_c301, &cb));
    assert_eq!(s, RVG_OK);
    let mode = |flags| RvgDisplayMode {
        width: 640,
        height: 480,
        flags,
        refresh_rate_hz: 60.0,
    };
    let modes = [
        mode(0),
        mode(RVG_DISPLAY_MODE_EDR),
        mode(RVG_DISPLAY_MODE_10BPC),
        mode(RVG_DISPLAY_MODE_EDR | RVG_DISPLAY_MODE_10BPC),
        mode(0x100),
    ];
    // SAFETY: the display made above; five modes.
    unsafe { rvg_display_set_modes(d, modes.as_ptr(), modes.len()) };
    // What the device wrote through its mapping of the page, in the guest's RAM.
    rig.p.fake_vm().sync();
    let page = 0x10u64 << APV_PAGE_SHIFT;
    let ram = rig._host.ram();
    let count = ram.read_vec(page + sp::MODE_LIST_COUNT, 2).unwrap();
    assert_eq!(u16::from_le_bytes([count[0], count[1]]), 5);
    let flags: Vec<u8> = (0..5)
        .map(|i| {
            ram.read_vec(page + sp::mode_entry(i) + sp::MODE_FLAGS, 1)
                .unwrap()[0]
        })
        .collect();
    assert_eq!(flags, [0, 1, 2, 3, 0]);
    // SAFETY: the display's reference.
    unsafe { rvg_display_release(d) };
    assert_eq!(rec.wait_destroys(1).len(), 1);
}

/// Each refusal in the header's order ends the table before the create returns; a config the
/// create cannot read runs no destroy
#[test]
fn refusals_end_the_table() {
    let rig = Rig::new(1);
    let rec = Rec::new();
    let cb = table(&rec);
    let (s, d) = create(rig.device, &config(0, 0x5eed_c101, &cb));
    assert_eq!(s, RVG_OK);
    let other = Rec::new();
    let ocb = table(&other);
    let mut c = config(0, 0x5eed_c102, &ocb);
    c.struct_size = 4;
    assert_eq!(create(rig.device, &c).0, RVG_ERR_INVALID_ARGUMENT);
    c.struct_size = sized::size_u32::<RvgDisplayConfig>();
    assert!(other.destroys().is_empty());
    assert_eq!(create(std::ptr::null_mut(), &c).0, RVG_ERR_INVALID_ARGUMENT);
    c.port = 1;
    assert_eq!(create(rig.device, &c).0, RVG_ERR_PORT_OUT_OF_RANGE);
    c.port = 0;
    c.serial_number = 0x5eed_c101;
    assert_eq!(create(rig.device, &c).0, RVG_ERR_SERIAL_IN_USE);
    c.serial_number = 0x5eed_c102;
    assert_eq!(create(rig.device, &c).0, RVG_ERR_PORT_IN_USE);
    let me = std::thread::current().id();
    assert_eq!(other.destroys(), vec![me; 4]);
    // SAFETY: the display made above.
    unsafe { rvg_display_release(d) };
    // The display's table ends once, maybe on the display thread after the release returned.
    assert_eq!(rec.wait_destroys(1).len(), 1);
}

/// Display A's cursor move holds the display thread until the child says go.
static MOVE_STARTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static MOVE_GO: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

unsafe extern "C-unwind" fn t_blocking_move(_ctx: *mut c_void) {
    use std::sync::atomic::Ordering::SeqCst;
    MOVE_STARTED.store(true, SeqCst);
    for _ in 0..1000 {
        if MOVE_GO.load(SeqCst) {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

/// Waits up to 10 s for `flag`.
fn wait_for(flag: &std::sync::atomic::AtomicBool) -> bool {
    for _ in 0..1000 {
        if flag.load(std::sync::atomic::Ordering::SeqCst) {
            return true;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    false
}

/// The child of [`a_display_released_with_a_job_queued_dies`]: display A's cursor move holds
/// the display thread; the guest rings display B's doorbell, whose cursor move is queued behind
/// it; the host releases B; the job then finds B's handler gone.
#[test]
#[ignore = "run by a_display_released_with_a_job_queued_dies in a child process"]
fn child_a_display_released_with_a_job_queued() {
    reims_vgpu_core::testkit::capture::set_capture_enabled(false);
    let rig = Rig::new(2);
    // The guest's root node, then a guest display on each port over its own page.
    // SAFETY: the rig's live device.
    unsafe { crate::device::rvg_device_mmio_write(rig.device, 0x101c, 1) };
    // SAFETY: as above.
    let core = unsafe { &*rig.device }.device.core();
    core.set_shared_display_state_page(0x10, 0, false).unwrap();
    core.set_shared_display_state_page(0x11, 1, false).unwrap();
    let rec = Rec::new();
    let mut a_cb = table(&rec);
    a_cb.cursor_move = Some(t_blocking_move);
    let (s, _a) = create(rig.device, &config(0, 0x5eed_c201, &a_cb));
    assert_eq!(s, RVG_OK);
    let b_cb = table(&rec);
    let (s, b) = create(rig.device, &config(1, 0x5eed_c202, &b_cb));
    assert_eq!(s, RVG_OK);
    // SAFETY: the rig's live device; the displays made above.
    unsafe {
        crate::device::rvg_device_mmio_write(rig.device, 0x1220, 0);
        assert!(wait_for(&MOVE_STARTED));
        crate::device::rvg_device_mmio_write(rig.device, 0x1220, 1);
        rvg_display_release(b);
    }
    eprintln!("RELEASED");
    MOVE_GO.store(true, std::sync::atomic::Ordering::SeqCst);
    std::thread::sleep(std::time::Duration::from_secs(5));
    eprintln!("SURVIVED");
}

/// A display released with one of its callback jobs queued on the display thread dies at that
/// job, as it did on the host's queue: the job reads the handler when it runs
#[test]
fn a_display_released_with_a_job_queued_dies() {
    use std::os::unix::process::ExitStatusExt as _;
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--ignored",
            "--exact",
            "display::tests::child_a_display_released_with_a_job_queued",
            "--nocapture",
            "--test-threads=1",
        ])
        .output()
        .unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.signal(), Some(6), "{out:?}");
    assert!(err.contains("RELEASED"), "{err}");
    assert!(
        err.contains("guest_display.cursor.move_handler_gone"),
        "{err}"
    );
    assert!(!err.contains("SURVIVED"), "{err}");
}

/// Reporting sleep on a headless display leaves guest/device state and callbacks unchanged.
#[test]
fn headless_window_sleep_is_a_host_only_noop() {
    let rig = Rig::new(1);
    let rec = Rec::new();
    let cb = table(&rec);
    let (status, display) = create(rig.device, &config(0, 0x5eed_c004, &cb));
    assert_eq!(status, RVG_OK);
    // SAFETY: the display is live until released below; this rig uses the fake platform.
    unsafe {
        rvg_display_set_window_suspended(display, true);
        rvg_display_set_window_suspended(display, false);
        rvg_display_set_window_suspended(display, false);
        let core = (&*display).display.core();
        assert_eq!(core.serial_num(), 0x5eed_c004);
        assert!(core.mode_list().is_empty());
        assert!(rec.lines().is_empty());
        rvg_display_release(display);
    }
    assert_eq!(rec.wait_destroys(1).len(), 1);
}
