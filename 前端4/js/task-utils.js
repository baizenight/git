/**
 * 返回出现在数组 a 中、但没有出现在数组 b 中的任务。
 *
 * @param {string[]} a
 * @param {string[]} b
 * @returns {string[]}
 */
function difference(a, b) {
  // TODO 目标一：找出差异任务
  const result = [];

  a.forEach((task) => {
    const notInB = b.indexOf(task) === -1;
    const notInResult = result.indexOf(task) === -1;

    if (notInB && notInResult) {
      result.push(task);
    }
  });

  return result;
}

/**
 * 返回所有任务清单中共同包含的任务。
 *
 * @param {string[][]} taskLists 由多份任务数组组成的数组
 * @returns {string[]}
 */
function intersection(taskLists) {
  // TODO 目标二：找出共同任务
  // 想一想：taskLists 为空时应该返回什么？
  if (taskLists.length === 0) {
    return [];
  }

  const firstList = unique(taskLists[0]);

  return firstList.filter((task) =>
    taskLists.every((list) => list.indexOf(task) !== -1)
  );
}



/**
 * 合并多份任务清单，并去除重复任务。
 *
 * @param {string[][]} taskLists 由多份任务数组组成的数组
 * @returns {string[]}
 */
function union(taskLists) {
  // TODO 目标三：汇总所有任务
  const result = [];

  taskLists.forEach((list) => {
    list.forEach((task) => {
      if (result.indexOf(task) === -1) {
        result.push(task);
      }
    });
  });

  return result;
}

function unique(tasks) {
  const result = [];

  tasks.forEach((task) => {
    if (result.indexOf(task) === -1) {
      result.push(task);
    }
  });

  return result;
}

// 将函数提供给页面中的index.js使用。
// 这一部分已经写好，不需要修改。
window.TaskUtils = {
  difference,
  intersection,
  union,
};
