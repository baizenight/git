(function () {
  'use strict';

  var links = Array.prototype.slice.call(
    document.querySelectorAll('.sidebar_nav a[href^="#"]')
  );
  if (!links.length) return;

  var activeLink = null;

  function setActive(id) {
    var next = document.querySelector('.sidebar_nav a[href="#' + id + '"]');
    if (!next || next === activeLink) return;
    if (activeLink) activeLink.classList.remove('is-active');
    next.classList.add('is-active');
    activeLink = next;
  }

  var targets = links
    .map(function (a) { return document.getElementById(a.getAttribute('href').slice(1)); })
    .filter(Boolean);

  // 用 IntersectionObserver 监听文章是否进入视口
  var observer = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (entry.isIntersecting) setActive(entry.target.id);
    });
  }, {
    rootMargin: '-80px 0px -60% 0px', // 顶部避开 sticky 顶栏
    threshold: 0
  });

  targets.forEach(function (el) { observer.observe(el); });

  // 页面加载时如果 URL 带 #article-x，同步高亮
  if (location.hash) {
    var id = location.hash.slice(1);
    if (document.getElementById(id)) setActive(id);
  }
})();