(function () {
  'use strict';

  var root = document.documentElement;
  var btn = document.getElementById('theme_changer');
  var STORAGE_KEY = 'blog-theme';

  // 读取上次的选择；没有就用系统偏好
  var saved = null;
  try {
    saved = localStorage.getItem(STORAGE_KEY);
  } catch (e) { /* file:// 或隐私模式下可能不可用，忽略 */ }

  if (saved === 'dark' || saved === 'light') {
    root.setAttribute('data-theme', saved);
  } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
    root.setAttribute('data-theme', 'dark');
  }

  function syncButton() {
    var isDark = root.getAttribute('data-theme') === 'dark';
    btn.textContent = isDark ? '切换浅色' : '切换深色';
    btn.setAttribute('aria-pressed', String(isDark));
    btn.setAttribute('title', isDark ? '切换到浅色主题' : '切换到深色主题');
  }

  btn.addEventListener('click', function () {
    var isDark = root.getAttribute('data-theme') === 'dark';
    var next = isDark ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    try {
      localStorage.setItem(STORAGE_KEY, next);
    } catch (e) { /* 忽略存储失败 */ }
    syncButton();
  });

  syncButton();
})();
