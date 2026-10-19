(function () {
  'use strict';

  var KEY = 'blog-messages';

  var form = document.getElementById('board_form');
  var list = document.getElementById('board_list');
  var nameEl = document.getElementById('board_name');
  var textEl = document.getElementById('board_text');
  if (!form || !list) return;

  function load() {
    try {
      return JSON.parse(localStorage.getItem(KEY)) || [];
    } catch (e) {
      return [];
    }
  }

  function save(items) {
    try {
      localStorage.setItem(KEY, JSON.stringify(items.slice(0, 50)));
    } catch (e) {
      /* 隐私模式或空间不足时忽略 */
    }
  }

  function formatTime(ts) {
    var d = new Date(ts);
    if (isNaN(d.getTime())) return '';
    var diff = (Date.now() - d.getTime()) / 1000;
    if (diff < 60) return '刚刚';
    if (diff < 3600) return Math.floor(diff / 60) + ' 分钟前';
    if (diff < 86400) return Math.floor(diff / 3600) + ' 小时前';
    return (d.getMonth() + 1) + '月' + d.getDate() + '日';
  }

  function render() {
    var items = load();
    list.textContent = '';

    if (!items.length) {
      var empty = document.createElement('li');
      empty.className = 'board_empty';
      empty.textContent = '还没有留言，来写第一条吧～';
      list.appendChild(empty);
      return;
    }

    items.forEach(function (item) {
      var li = document.createElement('li');
      li.className = 'board_item';

      var meta = document.createElement('div');
      meta.className = 'board_meta';

      var name = document.createElement('span');
      name.className = 'board_name';
      name.textContent = item.name || '匿名';

      var time = document.createElement('time');
      time.className = 'board_time';
      time.textContent = formatTime(item.time);

      meta.appendChild(name);
      meta.appendChild(time);

      var body = document.createElement('p');
      body.className = 'board_body';
      body.textContent = item.text;   // 用 textContent，天然防 XSS

      li.appendChild(meta);
      li.appendChild(body);
      list.appendChild(li);
    });
  }

  form.addEventListener('submit', function (e) {
    e.preventDefault();

    var text = textEl.value.trim();
    if (!text) return;

    var items = load();
    items.unshift({
      name: nameEl.value.trim().slice(0, 12),
      text: text.slice(0, 200),
      time: Date.now()
    });

    save(items);
    form.reset();
    render();
    textEl.focus();
  });

  render();
})();