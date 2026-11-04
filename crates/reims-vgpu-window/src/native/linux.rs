//! The window on Linux: SDL3's, on Wayland or X11, drawn through Vulkan. The window's shell and
//! cursor are SDL's shared ones (`crate::sdl`), and what Linux does its own way is here: SDL runs
//! on the library's UI thread (`ui`), where the window opens, changes and ends, as `sdl` places,
//! sizes and ends it, and its events become the host's input (`input`, `keymap`); each window's
//! render thread draws into its swapchain (`render`). This file is what the shared window code
//! asks of it: what draws into the window ([`FrameSink`]), the window's opening, changes and end,
//! posted to the UI thread, and the host's screens as SDL lists them.

mod drop_files;
mod input;
mod keymap;
mod render;
pub(crate) mod sdl;
mod ui;

use std::sync::Arc;

pub use render::FrameSink;
pub(crate) use render::Surface;

use crate::handle::Window;
use crate::screens::HostScreen;
use crate::sdl::shell;

/// Starts `window`'s render thread and posts its opening to the UI thread, starting that where it
/// has not started.
pub(crate) fn open(window: &Arc<Window>) {
    render::start(Arc::clone(window));
    let window = Arc::downgrade(window);
    ui::post(move || {
        if let Some(window) = window.upgrade() {
            shell::open(&window);
        }
    });
}

/// Posts to the UI thread that the guest of window `id` set a mode.
pub(crate) fn guest_mode_changed(id: u64) {
    ui::post(move || shell::guest_mode_changed(id));
}

/// Posts a cursor refresh of window `id` to the UI thread.
pub(crate) fn cursor_changed(id: u64) {
    ui::post(move || shell::cursor_changed(id));
}

/// Posts the host-reported guest sleep state to the UI thread.
pub(crate) fn suspended_changed(id: u64, suspended: bool) {
    ui::post(move || shell::suspended_changed(id, suspended));
}

/// Posts the close of window `id` to the UI thread.
pub(crate) fn close(id: u64) {
    ui::post(move || shell::close(id));
}

/// The host's screens, as SDL lists them (`ui`).
pub(crate) fn host_screens() -> Vec<HostScreen> {
    ui::screens()
}
