const members = [
  {
    name: '小鱼',
    role: '前端工程师',
    tasks: ['页面开发', '组件封装', '接口联调', '页面走查', '缺陷修复'],
  },
  {
    name: '小霜',
    role: '后端工程师',
    tasks: ['数据库设计', '接口开发', '接口联调', '需求验收', '缺陷修复'],
  },
  {
    name: '雨捷',
    role: 'UI设计师',
    tasks: ['视觉设计', '图标绘制', '页面走查', '交互验收'],
  },
  {
    name: '青子',
    role: '产品经理',
    tasks: ['需求分析', '原型设计', '需求评审', '需求验收', '版本复盘'],
  },
  {
    name: '微光娘',
    role: '测试工程师',
    tasks: ['测试用例', '接口联调', '页面走查', '需求验收', '缺陷修复'],
  },
];

const cooperationGroups = [
  ['前端工程师', '后端工程师', '测试工程师'],
  ['前端工程师', 'UI设计师', '测试工程师'],
  ['后端工程师', '产品经理', '测试工程师'],
];

const memberTaskContainer = document.getElementById('member-task-container');
const independentTaskContainer = document.getElementById('independent-task-container');
const cooperationTaskContainer = document.getElementById('cooperation-task-container');
const allTaskContainer = document.getElementById('all-task-container');
const allTaskCount = document.getElementById('all-task-count');

function createTaskList(tasks) {
  if (tasks.length === 0) {
    const empty = document.createElement('p');
    empty.className = 'empty-message';
    empty.textContent = '暂时没有结果，请检查对应的 TODO。';
    return empty;
  }

  const list = document.createElement('ul');
  list.className = 'task-list';

  tasks.forEach((task) => {
    const item = document.createElement('li');
    item.textContent = task;
    list.appendChild(item);
  });

  return list;
}

function createMemberCard(member, tasks) {
  const card = document.createElement('article');
  card.className = 'task-card';

  const title = document.createElement('h3');
  title.textContent = member.name;

  const role = document.createElement('span');
  role.className = 'role';
  role.textContent = member.role;

  card.appendChild(title);
  card.appendChild(role);
  card.appendChild(createTaskList(tasks));

  return card;
}

function createCooperationCard(roles, tasks) {
  const card = document.createElement('article');
  card.className = 'task-card';

  const title = document.createElement('h3');
  title.textContent = roles.join(' + ');

  card.appendChild(title);
  card.appendChild(createTaskList(tasks));
  return card;
}

function render() {
  const { difference, intersection, union } = window.TaskUtils;
  const allTaskLists = members.map((member) => member.tasks);

  members.forEach((member) => {
    memberTaskContainer.appendChild(createMemberCard(member, member.tasks));

    const otherTaskLists = members
      .filter((otherMember) => otherMember !== member)
      .map((otherMember) => otherMember.tasks);

    const tasksOwnedByOthers = union(otherTaskLists);
    const independentTasks = difference(member.tasks, tasksOwnedByOthers);

    independentTaskContainer.appendChild(
      createMemberCard(member, independentTasks)
    );
  });

  cooperationGroups.forEach((roles) => {
    const taskLists = roles.map((role) => {
      const member = members.find((item) => item.role === role);
      return member ? member.tasks : [];
    });

    const commonTasks = intersection(taskLists);
    cooperationTaskContainer.appendChild(
      createCooperationCard(roles, commonTasks)
    );
  });

  const allTasks = union(allTaskLists);
  allTaskCount.textContent = `共整理出 ${allTasks.length} 项不同任务`;

  if (allTasks.length === 0) {
    allTaskContainer.appendChild(createTaskList([]));
    return;
  }

  allTasks.forEach((task) => {
    const tag = document.createElement('span');
    tag.className = 'task-tag';
    tag.textContent = task;
    allTaskContainer.appendChild(tag);
  });
}

try {
  render();
} catch (error) {
  const errorBox = document.createElement('div');
  errorBox.className = 'error-box';
  errorBox.textContent = `页面运行时出现了问题：\n${error.message}\n\n可以打开开发者工具的 Console 查看更完整的信息。`;
  document.getElementById('root').appendChild(errorBox);
  console.error(error);
}
