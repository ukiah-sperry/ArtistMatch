/* ArtistMatch — main.js */

(function () {
  'use strict';

  // ── Flash dismiss (all pages) ────────────────────────────────
  document.querySelectorAll('.flash-dismiss').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var flash = btn.closest('.flash');
      if (flash) flash.remove();
    });
  });

  // ── Upload page ──────────────────────────────────────────────
  var zone   = document.querySelector('.upload-zone');
  var input  = zone && zone.querySelector('input[type="file"]');
  var submit = document.querySelector('[data-submit]');

  if (zone && input) {
    var preview = document.getElementById('upload-preview');
    var thumb   = document.getElementById('preview-thumb');
    var nameEl  = document.getElementById('preview-name');

    function updatePreview(file) {
      if (!preview) return;
      nameEl.textContent = file.name;
      preview.classList.add('visible');

      if (thumb && file.type.startsWith('image/')) {
        var reader = new FileReader();
        reader.onload = function (e) {
          thumb.src = e.target.result;
          thumb.style.display = 'block';
        };
        reader.readAsDataURL(file);
      } else if (thumb) {
        thumb.style.display = 'none';
      }
    }

    input.addEventListener('change', function () {
      var file = input.files[0];
      if (!file) return;
      updatePreview(file);
      if (submit) submit.disabled = false;
    });

    zone.addEventListener('dragover', function (e) {
      e.preventDefault();
      zone.classList.add('drag-over');
    });

    zone.addEventListener('dragleave', function (e) {
      if (!zone.contains(e.relatedTarget)) {
        zone.classList.remove('drag-over');
      }
    });

    zone.addEventListener('drop', function (e) {
      e.preventDefault();
      zone.classList.remove('drag-over');

      var file = e.dataTransfer.files[0];
      if (!file) return;

      var dt = new DataTransfer();
      dt.items.add(file);
      input.files = dt.files;

      updatePreview(file);
      if (submit) submit.disabled = false;
    });
  }

  // ── Result page: Create Playlist ─────────────────────────────
  var playlistBtn    = document.getElementById('create-playlist-btn');
  var playlistResult = document.getElementById('playlist-result');

  if (playlistBtn) {
    playlistBtn.addEventListener('click', function () {
      playlistBtn.disabled = true;
      playlistBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Creating playlist…';

      fetch('/create_playlist', { method: 'POST' })
        .then(function (res) {
          return res.json().then(function (data) {
            return { status: res.status, data: data };
          });
        })
        .then(function (payload) {
          var status = payload.status;
          var data   = payload.data;

          if (status === 403 && data.error === 'playlist_scope_required') {
            window.location.href = '/login?extended=1';
            return;
          }

          if (data.success) {
            playlistBtn.style.display = 'none';
            playlistResult.innerHTML =
              '<a class="btn btn-accent" href="' + data.playlist_url +
              '" target="_blank" rel="noopener noreferrer">' +
              '<i class="fab fa-spotify"></i> Open Playlist on Spotify →</a>';
          } else {
            playlistBtn.disabled = false;
            playlistBtn.innerHTML = '<i class="fab fa-spotify"></i> Create Spotify Playlist';
            playlistResult.innerHTML =
              '<span class="playlist-error"><i class="fas fa-exclamation-circle"></i> ' +
              (data.error || 'Something went wrong.') + '</span>';
          }
        })
        .catch(function () {
          playlistBtn.disabled = false;
          playlistBtn.innerHTML = '<i class="fab fa-spotify"></i> Create Spotify Playlist';
          playlistResult.innerHTML =
            '<span class="playlist-error">' +
            '<i class="fas fa-exclamation-circle"></i> Network error. Try again.</span>';
        });
    });
  }

}());
