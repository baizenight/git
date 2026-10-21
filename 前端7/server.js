const http = require('http');
const fs = require('fs');
const path = require('path');
const resources = require('./data/resources');

const PORT = 3000;

/** 返回 JSON 数据 */
function sendJson(response, statusCode, data) {
  response.statusCode = statusCode;
  response.setHeader('Content-Type', 'application/json; charset=utf-8');
  response.end(JSON.stringify(data));
}

/** 返回 public 文件夹中的静态文件。本题不要求修改。 */
function servePublicFile(pathname, response) {
  const fileMap = {
    '/': 'index.html',
    '/style.css': 'style.css',
    '/index.js': 'index.js',
  };
  const filename = fileMap[pathname];
  if (!filename) return false;

  const filePath = path.join(__dirname, 'public', filename);
  const extension = path.extname(filename);
  const contentTypes = {
    '.html': 'text/html; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.js': 'application/javascript; charset=utf-8',
  };

  fs.readFile(filePath, (error, content) => {
    if (error) {
      sendJson(response, 500, { code: 500, message: '服务器读取页面文件失败' });
      return;
    }
    response.statusCode = 200;
    response.setHeader('Content-Type', contentTypes[extension] || 'text/plain; charset=utf-8');
    response.end(content);
  });
  return true;
}

const server = http.createServer((request, response) => {
  const baseUrl = `http://${request.headers.host || `localhost:${PORT}`}`;
  const requestUrl = new URL(request.url, baseUrl);
  const pathname = requestUrl.pathname;

  console.log(`${request.method} ${pathname}`);

  if (request.method !== 'GET') {
    sendJson(response, 405, { code: 405, message: '本题暂时只支持 GET 请求' });
    return;
  }

  // TODO 1：完成 GET /api/studio
  if (pathname === '/api/studio') {
    // 提示：调用 sendJson(response, 状态码, 要返回的数据)
    sendJson(response, 200, {'code': 200, 'massage': "请求成功",'data': {'name': "微光工作室", 'direction': "前端",'description': "xxxxxxxxx"}})
    return;
  }

  // TODO 2：完成 GET /api/resources
  if (pathname === '/api/resources') {
    // 提示：resources 已经是数组，不需要重新复制数据
    sendJson(response, 200, {'code': 200, 'message': "请求成功", resources})
    return;
  }

  // TODO 3：完成 GET /api/random-resource
  if (pathname === '/api/random-resource') {
    // 提示：计算随机下标，再通过 resources[下标] 取出一条资源
    const index = Math.floor(Math.random() * resources.length);
    sendJson(response, 200, resources[index]);
    return;
  }

  if (servePublicFile(pathname, response)) return;

  // TODO 4：为不存在的地址返回 404 JSON
  sendJson(response, 404, {
    code: 404,
    message: '您访问的接口不存在',
  });
});

server.listen(PORT, () => {
  console.log(`服务器已启动：http://localhost:${PORT}`);
});

module.exports = { server, sendJson, servePublicFile };
