const resourceList = document.getElementById('resource-list');
const statusText = document.getElementById('status');

function createResourceCard(resource) {
  const card = document.createElement('article');
  card.className = 'resource-card';

  const category = document.createElement('span');
  category.className = 'resource-card__category';
  category.textContent = resource.category;

  const title = document.createElement('h3');
  title.textContent = resource.title;

  const description = document.createElement('p');
  description.textContent = resource.description;

  const link = document.createElement('a');
  link.href = resource.url;
  link.target = '_blank';
  link.rel = 'noopener noreferrer';
  link.textContent = '访问资源 →';

  card.append(category, title, description, link);
  return card;
}

async function loadResources() {
  try {
    // TODO 1：使用 fetch 请求 /api/resources
    const response = await fetch('/api/resources');

    // TODO 2：将响应转换为 JavaScript 数据
    const data = await response.json();

    // TODO 3：清空 resourceList，并遍历返回的资源
    // 每条资源可使用 createResourceCard(resource) 创建卡片
    resourceList.innerHTML = '';
    data.resources.forEach((resource) => {
      resourceList.append(createResourceCard(resource));
    });

    // TODO 4：更新状态文字，例如“共加载 5 条资源”
    statusText.textContent = `共加载 ${data.resources.length} 条资源`;
  } catch (error) {
    console.error('加载资源失败：', error);
    statusText.textContent = '资源加载失败';
    statusText.classList.add('error');
  }
}

loadResources();
