const KEY = 'SWJEEWD_M17tT3zGk';
const API = 'https://api.seniverse.com/v3/weather/now.json';
const CITIES = ['北京', '上海', '广州', '深圳', '成都', '杭州', '西安'];
const ORDER = ['晴', '多云', '阴', '雾', '小雨', '中雨', '大雨', '暴雨', '雷阵雨', '阵雨'];

let sort = 'asc';
let weather = CITIES.map((name) => ({ name, weather: '加载中', temperature: null }));

const list = document.getElementById('city-list');
const buttons = document.querySelectorAll('[data-sort]');
const show = (temperature) => (temperature == null ? '--' : `${temperature}°`);
const rank = (text) => {
  const index = ORDER.indexOf(text);
  return index === -1 ? ORDER.length : index;
};

function render() {
  const direction = sort === 'asc' ? 1 : -1;
  const cities = [...weather].sort((a, b) => {
    const difference = (a.temperature ?? 0) - (b.temperature ?? 0);

    return difference !== 0
      ? difference * direction
      : (rank(a.weather) - rank(b.weather)) * direction;
  });

  list.innerHTML = cities
    .map(
      (city) =>
        `<li class="city-row"><span class="city-name">${city.name}</span><span class="city-weather">${city.weather}</span><span class="city-temp">${show(city.temperature)}</span></li>`
    )
    .join('');

  buttons.forEach((button) =>
    button.setAttribute('aria-pressed', button.dataset.sort === sort)
  );
}

const getWeather = (name) =>
  fetch(`${API}?key=${KEY}&location=${encodeURIComponent(name)}`)
    .then((response) => response.json())
    .then((json) => {
      const now = json.results[0].now;
      return { name, weather: now.text, temperature: Number(now.temperature) };
    })
    .catch(() => ({ name, weather: '获取失败', temperature: null }));

async function load() {
  weather = await Promise.all(CITIES.map(getWeather));
  render();
}

buttons.forEach((button) =>
  button.addEventListener('click', () => {
    sort = button.dataset.sort;
    render();
  })
);

render();
load();
