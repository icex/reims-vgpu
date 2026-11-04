//! Opt-in file-drop transport. Only SDL drop events touch this socket; rendering never does.
//! One datagram carries one absolute native path, without quoting or line delimiters.

use std::io;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{FileTypeExt, MetadataExt};
use std::os::unix::net::UnixDatagram;
use std::path::Path;

/// A nonblocking connection to the desktop supervisor's private socket.
pub(super) struct DropFiles(UnixDatagram);

impl DropFiles {
    /// Unset means the existing window behaviour. Invalid endpoints are ignored with a warning.
    pub(super) fn from_environment() -> Option<Self> {
        let path = std::env::var_os("REIMS_VGPU_DROP_SOCKET")?;
        match Self::connect(Path::new(&path)) {
            Ok(socket) => Some(socket),
            Err(error) => {
                log::warn!(target: "reims_vgpu_window", "file drop unavailable: {error}");
                None
            }
        }
    }

    fn connect(path: &Path) -> io::Result<Self> {
        let parent = path.parent().ok_or(io::ErrorKind::InvalidInput)?;
        let directory = std::fs::symlink_metadata(parent)?;
        let endpoint = std::fs::symlink_metadata(path)?;
        if !path.is_absolute()
            || !directory.is_dir()
            || directory.mode() & 0o777 != 0o700
            || !endpoint.file_type().is_socket()
            || endpoint.uid() != directory.uid()
            || endpoint.mode() & 0o077 != 0
        {
            return Err(io::ErrorKind::PermissionDenied.into());
        }
        let socket = UnixDatagram::unbound()?;
        // The kernel also enforces the directory/socket permissions against this process.
        socket.connect(path)?;
        socket.set_nonblocking(true)?;
        Ok(Self(socket))
    }

    /// Never block SDL on a copy, a missing helper or a full receive queue.
    pub(super) fn send(&self, path: &[u8]) -> io::Result<()> {
        if !valid_path(path) {
            return Err(io::ErrorKind::InvalidInput.into());
        }
        self.0.send(path).map(|_| ())
    }
}

fn valid_path(bytes: &[u8]) -> bool {
    !bytes.is_empty()
        && bytes.len() <= 4096
        && !bytes.contains(&0)
        && Path::new(std::ffi::OsStr::from_bytes(bytes)).is_absolute()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn dropped_paths_are_native_bytes_not_shell_or_lines() {
        for path in [
            b"/tmp/a\nb".as_slice(),
            b"/tmp/'\"$(touch nope)",
            b"/tmp/\xff",
        ] {
            assert!(valid_path(path));
        }
        assert!(!valid_path(b"relative"));
        assert!(!valid_path(b"/nul\0name"));
        assert!(!valid_path(&vec![b'/'; 4097]));
    }

    #[test]
    fn drop_socket_is_private_and_preserves_a_whole_path() {
        let dir = std::env::temp_dir().join(format!("reims-drop-{}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).unwrap();
        let path = dir.join("drop.sock");
        let receiver = UnixDatagram::bind(&path).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        let sender = DropFiles::connect(&path).unwrap();
        let payload = b"/tmp/a\n'$(no shell)'";
        sender.send(payload).unwrap();
        let mut buffer = [0; 4096];
        let count = receiver.recv(&mut buffer).unwrap();
        assert_eq!(&buffer[..count], payload);
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).unwrap();
        assert!(DropFiles::connect(&path).is_err());
        std::fs::remove_file(path).unwrap();
        std::fs::remove_dir(dir).unwrap();
    }
}
