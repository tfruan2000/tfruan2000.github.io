---
title: 算法刷题笔记
author: tfruan
date: 2022-06-22 20:00:00 +0800
categories: [Coding]
tags: [Coding, Algorithm]
---

# 双指针

## 反转链表

1.LCR 141. 训练计划 III

给定一个头节点为 head 的单链表用于记录一系列核心肌群训练编号，请将该系列训练编号 倒序 记录于链表并返回。

```cpp
ListNode *now = head;
ListNode *res = nullptr;
while (now != nullptr) {
  ListNode *tmp = now->next;
  now->next= res;
  res = now;
  now = tmp;
}
```

## 快慢指针

1. LCR 140. 训练计划 II

给定一个头节点为 head 的链表用于记录一系列核心肌群训练项目编号，请查找并返回倒数第 cnt 个训练项目编号。

> 让快指针比慢指针快 `cnt` 个位即可
{: .prompt-info }

```cpp
ListNode* trainingPlan(ListNode* head, int cnt) {
    ListNode* low = head;
    ListNode* fast = head;
    for (int i = 0; i < cnt; ++i)
        fast = fast->next;
    while (fast != nullptr) {
        fast = fast->next;
        low = low->next;
    }
    return low;
}
```

## 滑动窗口

1.无重复的最长子串

```cpp
int lengthOfLongestSubstring(string s) {
  int size = s.size();
  if (size < 2) return size;
  unordered_map<char, int> map;
  int right = 0, left = 0;
  int max_len = 0;
  while (right < size) {
    char c = s[right];
    if (map.contains(c) && map[c] >= left) {
      left = map[c] + 1; // 移动
    }
    max_len = max(max_len, right - left + 1);
    ++right;
  }
  return max_len;
}
```

3.最小覆盖子串

```cpp
bool isEqual(unordered_map<char, int> &maps, unordered_map<char, int> &mapt) {
  for (auto it : mapt) {
    if (!maps.contains(it.first) || maps[it.first] < it.second)
      return false;
  }
  return true;
}
string minWindow(string s, string t) {
  if (s.size() < t.size()) return {};
  unordered_map<char, int> mapt, maps;
  for (char c : t) {
    ++mapt[c];
  }
  for (int i = 0; i < t.size(); ++i) {
    ++maps[s[i]];
  }
  int l = 0, r = t.size() - 1;
  int len = s.size() + 1, start = 0;
  while (r < s.size()) {
    // 右移直到覆盖
    while (r < s.size() && !isEqual(maps, mapt)) {
      ++r;
      if (r < s.size()) ++maps[s[r]];
    }
    // 左移直到不覆盖
    while (isEqual(maps, mapt)) {
      --maps[s[l]];
      ++l;
    }
    int tmp = l - 1;
    if (r - tmp + 1 < len) {
      len = r - tmp + 1;
      start = tmp;
    }
  }
  if (s.size() + 1 == len) return {};
  return s.substr(start, len);
}
```

3.LCR 183. 望远镜中最高的海拔

heights[i] 表示对应位置的海拔高度，返回[k, k + limit] 窗口内的 heights[i] 的最大值

```cpp
vector<int> maxAltitude(vector<int>& heights, int limit) {
    int n = heights.size();
    if (limit == 0 || n == 0)
        return {};
    if (n == 1)
        return heights;

    deque<int> q;
    vector<int> res;
    res.reserve(n - limit + 1);
    for (int i = 0; i < limit; ++i) {
        // 使用 while 一个个出队
        while (!q.empty() && heights[i] > q.back()) {
            // 如果 q.back() 大于当前值就不pop
            // 如果q中只有一个值，说明最大值一直在变
            q.pop_back();
        }
        q.push_back(heights[i]);
    }
    res.push_back(q.front());
    for (int i = limit; i < n; ++i) {
        while (!q.empty() && heights[i] > q.back()) {
            q.pop_back();
        }
        q.push_back(heights[i]);
        if (heights[i - limit] == q.front()) {
            // 当前记录的最大值应该被窗口排出
            q.pop_front();
        }
        res.push_back(q.front());
    }
    return res;
}
```

## 有序merge

1. LCR 142. 训练计划 IV

> 两个指针一起向前
{: .prompt-info }

```cpp
ListNode* trainningPlan(ListNode* l1, ListNode* l2) {
    if (l1 == nullptr)
        return l2;
    if (l2 == nullptr)
        return l1;
    ListNode res(-1); // 先初始化一个最小的node
    ListNode *curr = &res; // 使用指针引用，一边遍历一边修改
    ListNode *a = l1;
    ListNode *b = l2;
    // 归并排序的思想
    while (a != nullptr && b != nullptr) {
        if (a->val >= b->val) {
            curr->next = b;
            b = b->next;
        } else {
            curr->next = a;
            a = a->next;
        }
        curr = curr->next;
    }
    while (a != nullptr) {
        curr->next = a;
        a = a->next;
        curr = curr->next;
    }
    while (b != nullptr) {
        curr->next = b;
        b = b->next;
        curr = curr->next;
    }
    return res.next;
}
```

# 遍历

## 深度优先 DFS

1. LCR 130. 衣橱整理

```cpp
int dfs(vector<vector<bool>> &visited, int i, int j, int cnt, int m, int n) {
    auto digit = [](int k) -> int {
        int res = 0;
        while (k > 0) {
            res += k % 10;
            k = k / 10;
        }
        return res;
    };
    // 先判断取值是否合法
    if (i == m || j == n || visited[i][j] || digit(i) + digit(j) > cnt) return false;
    visited[i][j] = true;
    return 1 + dfs(visited, i + 1, j, cnt, m, n) + dfs(visited, i, j + 1, cnt, m, n);
}
int wardrobeFinishing(int m, int n, int cnt) {
    vector<bool> tmp(n, false);
    vector<vector<bool>> visited(m, tmp);
    // 向前遍历
    return dfs(visited, 0, 0, cnt, m, n);
}
```

2. LCR 153. 二叉树中和为目标值的路径

给你二叉树的根节点 root 和一个整数目标和 targetSum ，找出所有 从根节点到叶子节点 路径总和等于给定目标和的路径。

```cpp
vector<int> tmp;
vector<vector<int>> res;
// dfs(root, t) = dfs(root->left, t - root->val) + dfs(root->right, t - root->val)
void dfs(TreeNode *node, int t) {
    if (node == nullptr)
        return;
    // 尝试前进
    tmp.push_back(node->val);
    t -= node->val;
    if (node->left == nullptr && node->right == nullptr && t == 0) {
        res.push_back(tmp);
    }
    dfs(node->left, t);
    dfs(node->right, t);
    // 不行就退回
    tmp.pop_back();
}

vector<vector<int>> pathTarget(TreeNode* root, int target) {
    if (root == nullptr) return {};
    dfs(root, target);
    return res;
}
```

相似题：LeetCode437. 路径总和 III

不要求从root出发到叶子节点，只需要是从父到子

```cpp
int helper(TreeNode *curr, int target) {
  if (curr == nullptr) return 0;
  target -= curr->val;
  int res = helper(curr->left, target) + helper(curr->right, target);
  return target == 0 ? 1 + res : res;
}
int pathSum(TreeNode* root, int targetSum) {
  if (root == nullptr) return 0;
  return helper(root, targetSum) + pathSum(root->left, targetSum) + pathSum(root->right, targetSum);
}
```

相似题：LeetCode124. 二叉树中的最大路径和，每个节点有且只能使用一次

最初始的想法是先获得 preOrder，再使用滑动窗口，但是preOrder中相邻的元素不一定在树上有边的关系

```cpp
int res = INT_MIN;
int dfs(TreeNode *curr) {
  if (curr == nullptr) return 0;
  int l = max(curr->left, 0);
  int r = max(curr->right, 0);
  res = max(res, l + r + curr->val);
  return max(l + r) + curr->val; // 节点只能使用一次
}
int maxPathSum(TreeNode *root) {
  if (root == nullptr) return 0;
  dfs(root);
  return res;
}
```

3.LeetCode236. 二叉树的最近公共祖先

找 p 和 q 在 root 中的最近公共祖先

最简单的思想可以写一个dfs判断 p 和 q 分别在哪侧的子树上，但这样时间复杂度会高达 $O(n^2)$。换一种想法：往左右子树都找找

```cpp
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
  if (root == nullptr || root == p || root == q) return root;
  if (p == nullptr) return q;
  if (q == nullptr) return p;
  auto parent1 = lowestCommonAncestor(root->left, p, q);
  auto parent2 = lowestCommonAncestor(root->right, p, q);
  if (parent1 && parent2) return root;
  return parent1 != nullptr ? parent1 : parent2;
}
```

## 广度优化 BFS

层序遍历 即 广度优先搜索

1.LCR 149. 彩灯装饰记录 I

> 用 `queue` 实现层序遍历
{: .prompt-info }

一棵圣诞树记作根节点为 root 的二叉树，节点值为该位置装饰彩灯的颜色编号。请按照从 左 到 右 的顺序返回每一层彩灯编号。

输入：root = [8,17,21,18,null,null,6]
输出：[8,17,21,18,6]

```cpp
vector<int> decorateRecord(TreeNode* root) {
    if (root == nullptr) return {};
    vector<int> res;
    std::queue<TreeNode *> q;
    q.push(root);
    while (!q.empty()) {
        TreeNode* now = q.front();
        q.pop();
        if (now != nullptr) {
            res.push_back(now->val);
            q.push(now->left);
            q.push(now->right);
        }
    }
    return res;
}
```

2.LeetCode200. 岛屿数量

```cpp
int numIslands(vector<vector<char>>& grid) {
  int m = grid.size();
  if (m == 0) return 0;
  int n = grid[0].size();
  int res = 0;
  queue<std::pair<int, int>> q;
  vector<std::pair<int, int>> directions;
  directions.push_back({-1, 0});
  directions.push_back({1, 0});
  directions.push_back({0, -1});
  directions.push_back({0, 1});
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      if (grid[i][j] == '0')
        continue;
      ++res;
      q.push({i, j});
      grid[i][j] = '0';
      while (!q.empty()) {
        auto [x, y] = q.front();
        q.pop();
        for (auto [dx, dy] : directions) {
          int nx = x + dx;
          int ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= m || ny >=n || grid[nx][ny] == '0')
            continue;
          q.push({nx, ny});
          grid[nx][ny] = '0';
        }
      }
    }
  }
  return res;
}
```

3.LeetCode994. 腐烂的橘子

```cpp
int orangesRotting(vector<vector<int>>& grid) {
  int m = grid.size();
  if (m == 0) return -1;
  int n = grid[0].size();
  int good = 0;
  queue<std::pair<int, int>> bads;
  for (int i = 0; i < m; ++i) {
    for (int j =0; j < n; ++j) {
      int val = grid[i][j];
      if (val == 1) ++good;
      else if (val == 2) bads.push({i, j});
    }
  }
  int res = 0;
  vector<std::pair<int, int>> directions;
  directions.push_back({-1, 0});
  directions.push_back({1, 0});
  directions.push_back({0, -1});
  directions.push_back({0, 1});
  while (!bads.empty() && good > 0) {
    int size = bads.size(); // 每轮都把当前queue剩下的坏橘子访问一遍
    for (int i = 0; i < size; ++i) {
      auto [x, y] = bads.front();
      bads.pop();
      for (auto [dx, dy] : directions) {
        int nx = x + dx;
        int ny = y + dy;
        if (nx < 0 || nx >= m || ny < 0 || ny >= n || grid[nx][ny] != 1)
          continue;
        grid[nx][ny] = 2;
        bads.push({nx, ny});
        --good;
      }
    }
    ++res;
  }
  return good == 0 ? res : -1;
}
```

4.LeetCode207. 课程表

你这个学期必须选修 numCourses 门课程，记为 0 到 numCourses - 1 。

在选修某些课程之前需要一些先修课程。 先修课程按数组 prerequisites 给出，其中 prerequisites[i] = [ai, bi] ，表示如果要学习课程 ai 则 必须 先学习课程  bi 。

```cpp
vector<int> visit;
bool dfs(int v, vector<vector<int>> &g) {
  if (g[v].size() == 0) return true;
  if (visit[v] == -1) return true;
  if (visit[v] == 1) return false;
  visit[v] = 1; // 防止有环
  auto needs = g[v];
  for (int need : needs) {
    bool res = dfs(need, g);
    if (!res) return false;
  }
  visit[v] = -1;
  return true;
}
bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
  vector<vector<int>> g(numCourses);
  visit = vector<int>(numCourses, 0);
  for (auto it = prerequisites.begin(); it != prerequisites.end(); ++it) {
    g[(*it)[0]].push_back((*it)[1]);
  }
  for (int i = 0; i < numCourses; ++i) {
    vector<int> needs = g[i]; // 上当前课程的前置需求
    if (needs.size() == 0) continue;
    for (int need : needs) {
      bool res = dfs(need, g); // 看 need 课程能不能上
      if (!res) return false;
    }
  }
  return true;
}
```

### 先序遍历

1. LCR 143. 子结构判断

给定两棵二叉树 tree1 和 tree2，判断 tree2 是否以 tree1 的某个节点为根的子树具有 相同的结构和节点值 。

```cpp
bool isSame(TreeNode *A, TreeNode *B) {
    if (B == nullptr) return true;
    if (A == nullptr) return false;
    return A->val == B->val && isSame(A->left, B->left) && isSame(A->right, B->right);
}
bool isSubStructure(TreeNode* A, TreeNode* B) {
    if (A == nullptr || B == nullptr) return false;
    // 判断 以A为根、以A->left为根、以A->right为根
    return isSame(A, B) || isSubStructure(A->left, B) || isSubStructure(A->right, B);
```

### 中序遍历

1. LCR 155. 将二叉搜索树转化为排序的双向链表

将一个 二叉搜索树 就地转化为一个 已排序的双向循环链表 。

```cpp
Node *pre;
Node *firstShit;
void dfs(Node *curr) {
    if (curr == nullptr) return;
    dfs(curr->left);
    if (pre) {
        // pre 作为 curr 的左侧
        curr->left = pre;
        pre->right = curr;
    }
    pre = curr;
    dfs(curr->right);
}
Node* treeToDoublyList(Node* root) {
    if (root == nullptr) return nullptr;
    firstShit = root;
    while (firstShit->left) {
        firstShit = firstShit->left;
    }
    pre = nullptr;
    dfs(root);
    // 把头和尾连起来
    pre->right = firstShit;
    firstShit->left = pre;
    return firstShit;
}
```

另一种解法：递归，认为 treeToDoublyList 返回一定是有序的

```cpp
Node* treeToDoublyList(Node* root) {
    if (root == nullptr) return nullptr;
    Node *firstShit = root;
    Node *lastShit = root;
    // treeToDoublyList 返回的一定是有序的
    Node *leftN = treeToDoublyList(root->left);
    Node *rightN = treeToDoublyList(root->right);
    if (leftN) {
        // 当前的 firstShit 应该直接接到 leftN 的后面
        leftN->left->right = firstShit;
        firstShit->left = leftN->left;
        firstShit = leftN;
    }
    if (rightN) {
        // 当前的 lastShit 应该直接接到 rightN 的前
        Node *tmp = rightN->left;
        lastShit->right = rightN;
        rightN->left = lastShit;
        lastShit = tmp;
    }
    firstShit->left = lastShit;
    lastShit->right = firstShit;
    return firstShit;
}
```

相似题：LeetCode 108 将有序数组转换为二叉搜索树

将左右树分开处理，使用分治的思想

```cpp
TreeNode *build(vector<int> &nums, int l, int r) {
  if (l > r) return nullptr;
  int m = (r - l) / 2 + l;
  TreeNode *res = new TreeNode(nums[m]);
  res->left = build(nums, l, m - 1);
  res->right = build(nums, m + 1, r);
  return res;
}
TreeNode *sortedArrayToBST(vector<int>& nums) {
    int size = nums.size();
    if (size == 0) return nullptr;
    return build(nums, 0, size - 1);
}
```

2. LCR 174. 寻找二叉搜索树中的目标节点

某公司组织架构以二叉搜索树形式记录，节点值为处于该职位的员工编号。请返回**第 cnt 大**的员工编号。（右根左）

```cpp
int count = 0;
int res = 0;
int findTargetNode(TreeNode* root, int cnt) {
    dfs(root, cnt);
    return res;
}
void dfs(TreeNode *curr, int cnt) {
    if (curr == nullptr) return;
    dfs(curr->right, cnt); // 先往右走到底，再count++
    ++count;
    if (count == cnt) res = curr->val;
    dfs(curr->left, cnt);
}
```

如果是返回**第 cnt 小**的员工编号，则是左根右

```cpp
void dfs(TreeNode *curr, int cnt) {
    if (curr == nullptr) return;
    dfs(curr->left, cnt); // 先往左走到底，再count++
    ++count;
    if (count == cnt) res = curr->val;
    dfs(curr->right, cnt);
}
```

# 分治与回溯

## 分治

将问题分为多个子问题，一段段解决再合并起来，最重要的是边界部分的处理。

1. LCR 152. 验证二叉搜索树的后序遍历序列

请实现一个函数来判断整数数组 postorder 是否为二叉搜索树的后序遍历结果。

树的后序遍历：左-右-根 -> 每个子问题为：验证左右子树是否满足对应条件。

```cpp
bool trave(vector<int> &postorder, int root, inst start) {
  // 此时遍历的只有root节点 || 此时遍历的只有root节点和另一个节点
  if (root == start || root - start == 1) return true;
  int left = start, right = root - 1;
  int rootVal = postorder[root];
  while (left < root && postorder[left] < rootVal) ++left;
  while (right >= start && postorder[right] > rootVal) --right;
  // 当以root为根后，剩下的部分已不存在左子/右子
  if ((left == start && right == start - 1) || (left == root && right == root - 1))
    return trave(postorder, root - 1, start);
  if (left == right + 1)
    return trave(postorder, left - 1, start) && trave(postorder, root - 1, right + 1);
  return false;
}

bool verifyTreeOrder(vector<int>& postorder) {
    int size = postorder.size();
    if (size == 0 || size == 1) return true;
    return trave(postorder, size - 1, 0);
}
```

2. LCR 164. 破解闯关密码

归并排序的思想，需要熟记归并排序的模版，根据问题设计比较函数 `cmp` 即可

```cpp
// 从大到小排序
bool cmp(int lhs, int rhs) {
  return lhs > rhs;
}

void merge(std::vector<int> &vec, int left, int mid, int right) {
  std::vector<int> tmp(right - left + 1);
  int i = left, j = mid + 1, k = 0;
  while (i <= mid && j <= right) {
    if (cmp(vec[i], vec[j])) {
      tmp[k++] = vec[i++];
    } else {
      tmp[k++] = vec[j++];
    }
  }
  while (i <= mid) {
    tmp[k++] = vec[i++];
  }
  while (j <= right) {
    tmp[k++] = vec[j++];
  }
  for (i = 0; i < k; ++i) {
    vec[left + i] = tmp[i];
  }
}

void merge_sort(std::vector<int> &vec, int left, int right) {
  if (left < right) {
    int mid = left + (right - left) / 2;
    merge_sort(vec, left, mid);
    merge_sort(vec, mid + 1, right);
    merge(vec, left, mid, right);
  }
}
```

再回到这道题，题目要求：

一个拥有密码所有元素的非负整数数组 password，密码是 password 中所有元素拼接后得到的最小的一个数，请编写一个程序返回这个密码。

- 输入: password = [0, 3, 30, 34, 5, 9]
- 输出: "03033459"

需要主要，比较函数的设置， 24 应该放在 2438 前，但是应该放在 2401 后面。这时候不妨转为 string，然后拼接比较。

> "242438" < "243824", "242401" > "240124"

这时候只需要重新设计比较函数如下，其他保持 memge_sort

```cpp
bool cmp(int lhs, int rhs) {
  return std::to_string(lhs) + std::to_string(rhs) < std::to_string(rhs) + std::to_string(lhs);
}
```

3.LeetCode 105 从先序和中序遍历来构建树

```cpp
void build(vector<int> &preorder, int &idx, vector<int> &inorder, int l, int r, TreeNode* &curr) {
  if (l < r || idx >= preorder.size()) return;
  int val = preorder[idx++];
  curr = new TreeNode(val);
  if (l < r) {
    auto iter= inorder.begin();
    int mid = std::find(iter + l, iter + r + 1, val);
    build(preorder, idx, inorder, l, mid - 1, curr->left);
    build(preorder, idx, inorder, mid + 1, r, curr->right)
  }
}
TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
    int size = preorder.size();
    if (size == 0) return nullptr;
    TreeNode *res = nullptr;
    int idx = 0;
    build(preorder, idx, inorder, 0, size - 1, res);
    return res;
}
```

## 二分

1.[LeetCode4. 寻找两个正序数组的中位数](https://leetcode.cn/problems/median-of-two-sorted-arrays/description/?envType=study-plan-v2&envId=top-100-liked)

给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。

算法的时间复杂度应该为 O(log (m+n)) 。

```cpp
// k : 合并后的第 k 个元素
int findVal(vector<int>& nums1, int i, vector<int>& nums2, int j, int k) {
  int size1 = nums1.size();
  int size2 = nums2.size();
  if (i == size1) return nums2[j + k - 1];
  if (j == size2) return nums1[i + k - 1];
  if (k == 1) return min(nums1[i], nums2[j]);
  int si = min(size1, i + k / 2);
  int sj = min(size2, j + k / 2);
  if (nums1[si - 1] < nums2[sj - 1]) {
    return findVal(nums1, si, nums2, j, k - (si - i));
  }
  return findVal(nums1, i, nums2, sj, k - (sj - j));
}
double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
  int len = nums1.size() + nums2.size();
  if (len % 2 == 0) {
    int l = findVal(nums1, 0, nums2, 0, len / 2);
    int r = findVal(nums1, 0, nums2, 0, len / 2 + 1);
    return (double) (l + r) / 2;
  }
  return (double) findVal(nums1, 0, nums2, 0, len / 2 + 1);
}
```

## 回溯

前进一步 + 函数 + 后退一步

1.LCR 129. 字母迷宫

字母迷宫游戏初始界面记作 m x n 二维字符串数组 grid，请判断玩家是否能在 grid 中找到目标单词 target。

注意：寻找单词时 必须 按照字母顺序，通过水平或垂直方向相邻的单元格内的字母构成，同时，同一个单元格内的字母 不允许被重复使用 。

```cpp
bool trave(vector<vector<char>>& grid, string target, vector<vector<bool>> &visit, int i, int j, int idx) {
  if (idx == target.size()) return true;
  if (i < 0; || i == grid.size() || j < 0 || j == grid[0].size() || visit[i][j]) return false;
  if (grid[i][j] != target[idx]) return false;
  visit[i][j] = true; // 前进
  if (trave(grid, target, visit, i + 1, j, idx) ||
      trave(grid, target, visit, i - 1, j, idx) ||
      trave(grid, target, visit, i, j + 1, idx) ||
      trave(grid, target, visit, i, j - 1, idx))
    return true;
  visit[i][j] = false; // 回退
  return false;
}
bool wordPuzzle(vector<vector<char>>& grid, string target) {
    int m = grid.size();
    int n = grid[0].size();
    vector<bool> tmp(n, false);
    vector<vector<bool>> visit(m, tmp);
    for (int i = 0; i < m; ++i) { // 可以从任意位置开始
        for (int j = 0; j < n; ++j) {
            if (trave(grid, target, visit, i, j, 0)) return true;
        }
    }
    return false;
}
```

2.LeetCode46. 全排列

```cpp
int size;
vector<bool> visit;
vector<int> tmp;
vector<vector<int>> res;
void trave(vector<int>& nums) {
  for (int i = 0; i < size; ++i) {
    if (visit[i]) continue;
    tmp.push_back(nums[i]);
    visit[i] = true;
    trave(nums);
    tmp.pop_back();
    visit[i] = false;
  }
  if (tmp.size() == size) {
    res.push_back(tmp);
  }
}
vector<vector<int>> permute(vector<int>& nums) {
  size = nums.size();
  if (size == 0) return {};
  if (size == 1) return {nums};
  visit = vector<bool>(size, false);
  trave(nums);
  return res;
}
```

3.LeetCode78. 子集

```cpp
int size;
vector<int> tmp;
vector<vector<int>> res;
void trave(vector<int> &nums, int begin) {
  if (begin > size) return;
  for (int i = begin; i < size; ++i) {
    tmp.push_back(nums[i]);
    res.push_back(tmp);
    trave(nums, i + 1);
    tmp.pop_back();
  }
}
vector<vector<int>> subsets(vector<int>& nums) {
  size = nums.size();
  if (size == 0) return {};
  if (size == 1) return {nums, {}};
  trave(nums, 0);
  res.push_back({});
  return {};
}
```

4.LeetCode39. 组合总和

```cpp
vector<vector<int>> res;
vector<int> tmp;
void dfs(vector<int> &candidants, int begin, int target, int curr) {
  if (target == curr) {
    res.push_back(tmp);
    return;
  }
  if (target < curr || begin >= candidants.size()) return;
  for (int i = begin; i < candidants.size(); ++i) {
    int val = candidants[i];
    if (curr + val > target) break;
    tmp.push_back(val);
    dfs(candidants, i, target, curr + val); // 可以重复使用当前元素
    tmp.pop_back();
  }
}
vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
  sort(candidates.begin(), candidates.end());
  dfs(candidants, 0, target, 0);
}
```

5.[LeetCode131.分割回文串](https://leetcode.cn/problems/palindrome-partitioning/description/)

```cpp
int n = 0;
vector<vector<string>> resVec;
vector<string> res;
bool isTarget(string &s, int l, int r) {
  while (l < r) {
    if (s[l] != s[r]) return false;
    ++l;
    --r;
  }
  return false;
}
void func(string &s, int idx) {
  if (idx == n) {
    resVec.push_back(res);
    return;
  }
  for (int len = 1; len <= n - idx; ++len) {
    if (isRight(s, idx, idx + len - 1)) {
      res.push_back(s.substr(idx, len));
      func(s, idx + len);
      res.pop_back();
    }
  }
}
vector<vector<string>> partition(string &s) {
  n = s.size();
  func(s, 0);
  return  resVec;
}
```

6.LeetCode 51 N皇后

将n个“皇后”放到n*n的棋牌上，要求互相不攻击。“皇后”会攻击同行、同列、同对角线上的其它棋子。

```cpp
vector<vector<string>> res;
vector<bool> checkCol, check1, check2;
int total;
bool isVaild(vector<string> &board, int row, int col) {
  return !(checkCol[col] || check1[total + row - col - 1] || check2[row + col]);
}
void search(vector<string> &board, int row) {
  if (row == total) {
    res.push_back(board);
    return;
  }
  for (int col = 0; col < total; ++col) {
    if (isVaild(board, row, col)) {
      board[row][col] = 'Q';
      checkCol[col] = check1[total + row - col - 1] = check2[row + col] = true;
      search(board, row + 1);
      board[row][col] = '.';
      checkCol[col] = check1[total + row - col - 1] = check2[row + col] = false;
    }
  }
}
vector<vector<string>> solveNQueens(int n) {
  vector<string> board(n, string(n, '.'));
  total = n;
  checkCol = vector<bool>(n, false);
  check1 = check2 = vector<bool>(2 * n - 1, false);
  search(board, 0);
  return res;
}
```

7.pdd笔试：有n个城市，n-1条城市之间的路，到一个城市收获开心值，走一条路掉开心值，求最大的开心值

```cpp
int getMaxB(vector<int> &an, unordered_map<int, unordered_map<int, int>> &map,
            vector<bool> &visited, vector<int> &dp) {
  if (dp[u] != -1) {
    return dp[u];
  }
  int maxB = 0;
  for (auto [v, w] : map[u]) {
    if (visited[v]) continue;
    visited[v] = true;
    int benefit = an[v] - w + getMaxB(an, map, visited, v, dp);
    maxB = max(maxB, benefit);
    visited[v] = false; // 回溯
  }
  dp[u] = maxB;
  return maxB;
}

int main() {
  int n;
  cin >> n;
  vector<int> an(n); // 到一个城市的开心值
  for (int i = 0; i < n; ++i) {
    cin >> an[i];
  }
  unordered_map<int, unordered_map<int, int>> map;
  for (int i = 0; i < n - 1; ++i) {
    int u, v, w;
    cin >> u >> v >> w;
    u--;
    v--;
    map[u][v] = w;
    map[v][u] = v;
  }
  int res = -1;
  // 遍历开始
  for (int i = 0; i < n; ++i) {
    vector<bool> visited(n, false);
    vector<int> dp(n, -1);
    visited[i] = true;
    int tmp = an[i] + getMaxB(an, map, visited, dp);
    res = max(res, tmp);
  }
  cout << res;
  return 0;
}
```

其实这种思想并不能过多少，应该优化为下面的代码

```cpp
int res = INT_MIN;
int dfs(int curr, int parent, vector<int> &an, unordered_map<int, unordered_map<int, int>> &map) {
  int maxB = an[curr];
  int max_path1 = 0, max_path2 = 0; // 到 curr 点的最长和次长
  for (auto [u, w] : map[curr]) {
    if (u == parent) continue;
    int subMax = dfs(u, curr, an, map) - w;
    if (subMax > max_path1) {
      max_path1, max_path2 = subMax, max_path1;
    } else if (subMax > max_path2) {
      max_path2 = subMax;
    }
    maxB += max_path1; // 从 curr 出发的最长
    res = max(res, maxB + max_path2);
    return maxB;
  }
}

int main() {
  int n;
  cin >> n;
  vector<int> an(n); // 到一个城市的开心值
  for (int i = 0; i < n; ++i) {
    cin >> an[i];
  }
  unordered_map<int, unordered_map<int, int>> map;
  for (int i = 0; i < n - 1; ++i) {
    int u, v, w;
    cin >> u >> v >> w;
    u--;
    v--;
    map[u][v] = w;
    map[v][u] = v;
  }
  dfs(0, -1, an, map);
  return res;
}
```

# 排序

## 基础模版

![sort](/assets/img/blog/img_coding_note/computational_complexity.png)

- 快排

每次挑出一个基准，比基准小的放左边，比基准大的放右边

```cpp
#include <iostream>
#include <vector>

void quickSort(std::vector<int>& arr, int left, int right) {
    if (left >= right) return; // 递归终止条件

    int pivot = arr[left]; // 选择最左边的元素作为基准
    int i = left + 1;
    int j = right;

    while (i <= j) {
        // 从左边找到一个大于基准的元素
        while (i <= j && arr[i] <= pivot) i++;
        // 从右边找到一个小于基准的元素
        while (i <= j && arr[j] >= pivot) j--;
        // 如果左指针还在右指针的左边，交换它们
        if (i < j) std::swap(arr[i], arr[j]);
    }

    // 将基准元素放到正确的位置
    std::swap(arr[left], arr[j]);

    // 对基准左边和右边的子数组递归排序
    quickSort(arr, left, j - 1);
    quickSort(arr, j + 1, right);
}
```

- 归并

```cpp
bool cmp(int lhs, int rhs) {
  return lhs > rhs;
}

void merge(std::vector<int> &vec, int left, int mid, int right) {
  std::vector<int> tmp(right - left + 1);
  int i = left, j = mid + 1, k = 0;
  // 左侧和右侧都先把符合条件的放入
  while (i <= mid && j <= right) {
    if (cmp(vec[i], vec[j])) {
      tmp[k++] = vec[i++];
    } else {
      tmp[k++] = vec[j++];
    }
  }
  while (i <= mid) tmp[k++] = vec[i++];
  while (j <= right) tmp[k++] = vec[j++];
  for (i = 0; i < k; ++i) vec[left + i] = tmp[i];
}

void merge_sort(std::vector<int> &vec, int left, int right) {
  if (left < right) {
    int mid = left + (right - left) / 2;
    merge_sort(vec, left, mid);
    merge_sort(vec, mid + 1, right);
    merge(vec, left, mid, right);
  }
}
```

- 冒泡排序：比较相邻的元素

```cpp
void bubble_sort(T arr[], const int len){
    for(int i=0; i<len; i++)
        for(int j=0; j<len-i-1; j++)
            // 每一轮最大的数都会被放到最后
            if(arr[j] > arr[j+1]){
                T tmp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = tmp;
            }
}
```

- 插入排序：每次选择一个，然后在它左侧的数组中找到位置，其左侧是有序

```cpp
template<typename T>
void insert_sort(T arr[], const int len){
    int j;
    T tmp;
    for(int i=1; i<len; i++){
        if(arr[i] < arr[i-1]){
            tmp = arr[i]; // 给arr[i]找到合适的位置
            for(j=i-1; j>=0 && arr[j]>tmp; j--)
                arr[j+1] = arr[j]; // 左移
        }
        arr[j+1] = tmp;
    }
}
```

- 选择排序：每次选出一个最小的值

```cpp
template<typename T>
void select_sort(T arr[], const int len){
    for(int i=0; i<len-1; i++){
        int min_index = i;
        for(int j=i+1; j<len; j++)
            if(arr[j] < arr[min_index])
                min_index = j;
        if(min_index != i){
            T tmp = arr[i];
            arr[i] = arr[min_index];
            arr[min_index] = tmp;
        }
    }
}
```

## 例题

1. LCR 159. 库存管理 III

仓库管理员以数组 stock 形式记录商品库存表，其中 stock[i] 表示对应商品库存余量。请返回库存余量最少的 cnt 个商品余量，返回 顺序不限。

即选出前n个最小的 -> 思路一：选择排序，每次选一个最小的，压入res； 思路二：快速排序，选择定好基准，左边都小于基准，右边都大于

- 选择排序

```cpp
vector<int> inventoryManagement(vector<int>& stock, int cnt) {
    int size = stock.size();
    if (size == 0 || cnt == size ) return stock;
    vector<int> res(cnt);
    // 选择排序
    for (int i = 0; i < cnt; ++i) {
        int min_idx = i;
        for (int j = i + 1; j < size ; ++j) {
            if (stock[min_idx] > stock[j]) min_idx = j;
        }
        res[i] = stock[min_idx];
        if (i != min_idx) {
            int tmp = stock[i];
            stock[i] = stock[min_idx];
            stock[min_idx] = tmp;
        }
    }
    return res;
}
```

- 快速排序

```cpp
vector<int> quick_sort(vector<int>& stock, int cnt, int start, int end) {
    // 快速排序
    int l = start, r = end;
    int base = start;
    while (l < r) {
        while (l < r && stock[base] <= stock[r]) r--;
        while (l < r && stock[base] >= stock[l]) l++;
        swap(stock[l], stock[r]);
    }
    swap(stock[l], stock[base]);
    // 一直找第cnt的位置
    if (cnt < l) return quick_sort(stock, cnt, start, l - 1);
    if (cnt > l) return quick_sort(stock, cnt, l + 1, end);
    vector<int> ans;
    ans.assign(stock.begin(), stock.begin() + cnt);
    return ans;
}
vector<int> inventoryManagement(vector<int>& stock, int cnt) {
    int size = stock.size();
    if (cnt == 0) return {};
    if (size == 0 || cnt == size) return stock;
    return quick_sort(stock, cnt, 0, size - 1);
}
```

# 动态规划问题

动归问题的关键是：

1.状态转移函数

2.dp初始化

以 LCR 166. 珠宝的最高价值 为例

现有一个记作二维矩阵 frame 的珠宝架，其中 frame[i][j] 为该位置珠宝的价值。拿取珠宝的规则为：

只能从架子的左上角开始拿珠宝 -> 从 dp[0][0] 出发

每次可以移动到右侧或下侧的相邻位置 -> dp[i][j] 只能转移到 dp[i+1][j] 或 dp[i+1][j]

到达珠宝架子的右下角时，停止拿取 -> 结束点为 dp[maxi][maxj]

注意：珠宝的价值都是大于 0 的。除非这个架子上没有任何珠宝，比如 frame = [[0]]。

```cpp
// 状态转移函数 res[i][j] = max(res[i-1][j], res[i][j-1]) + frame[i][j]
int jewelleryValue(vector<vector<int>>& frame) {
    if (frame.size() == 0) return 0;
    int maxi = frame.size();
    int maxj = frame[0].size();
    vector<vector<int>> res(maxi, vector<int>(maxj));
    res[0][0] = frame[0][0];
    for (int i = 0; i < maxi; ++i) {
        for (int j = 0; j < maxj; ++j) {
            if (i == 0) {
                if (j > 0) {
                    res[0][j] = res[0][j-1] + frame[0][j];
                }
                continue;
            }
            if (j == 0) {
                res[i][0] = res[i - 1][0] + frame[i][0];
                continue;
            }
            res[i][j] = std::max(res[i-1][j], res[i][j-1]) + frame[i][j];
        }
    }
    return res[maxi - 1][maxj - 1];
}
```

例：最大子数组和

转移函数 `dp[i] = max(dp[i - 1] + num[i], num[i])`

## 背包问题

dp[i][j]表示将前i种物品装进限容量为j的背包可以获得的最大价值, 0<=i<=N, 0<=j<=V

物体i最多有si[i]个，其体积是vi[i]，价值是wi[i]

状态转移方程

```cpp
for(int i=1; i<=N; i++){
 for(int j=1; j<=V; j++){
  dp[i][j] = ...
```

1. 不装入第i件物品，即`dp[i−1][j]`；
2. 装入第i件物品（前提是能装下）
    1. 01背包问题：si[i] = 1
    `dp[i][j] = max(dp[i−1][j], dp[i-1][j−vi[i]]+w[i]) // j >= v[i]`
    2. 完全背包问题：si[i] = 无穷
    `dp[i][j] = max(dp[i−1][j], dp[i][j−vi[i]]+w[i]) // j >= v[i]`
    3. 多重背包问题： si[i] 大于等于1，不等于无穷
    `k为装入第i种物品的件数, k <= min(n[i], j/w[i])`
    `dp[i][j] = max{(dp[i-1][j − k*w[i]] + k*v[i]) for every k}`

### 01背包问题

dp[i][j]表示将前i种物品装进限重为j的背包可以获得的最大价值,  0<=i<=N, 0<=j<=V

1. 不装入第i件物品，即`dp[i−1][j]`；
2. 装入第i件物品（前提是能装下），即`dp[i-1][j−vi[i]]+w[i]`

基础代码如下

```cpp
void pack_01(){
    vector<vector<int>>dp(N+1, vector<int>(V+1, 0));
    for(int i=1; i<=N; i++){
        for(int j=1; j<=V; j++){
            if(j >= vi[i])
                dp[i][j] = max(dp[i-1][j], dp[i-1][j-vi[i]] + wi[i]);
            else
                dp[i][j] = dp[i-1][j];
        }
    }
    cout << dp[N][V];
}
```

由于dp[i][j]只和上一轮的数据dp[i-1][j]相关，并不需要保留更前的数据，所以可以只用一维dp数组。j的遍历改为从大到小，可以防止j值较小时被覆盖（新的dp[j]需要使用旧的dp[j-1]计算）

因此可以化简为：

```cpp
void pack_01(){
    vector<int>dp(V+1);
    for(int i=1; i<=N; i++)
        for(int j=V; j>=vi[i]; j--)
            dp[j] = max(dp[j], dp[j-vi[i]]+wi[i]);
    cout << dp[V];
}
```

上述两个代码的数据初始化如下

```cpp
vector<int> vi(N+1, 0);
vector<int> wi(N+1, 0);
vi[0] = 0;
wi[0] = 0;
for(int i=1; i<=N; i++)
    cin >> vi[i] >> wi[i];
```

### 完全背包

dp[i][j]表示将前i种物品装进限重为j的背包可以获得的最大价值,  0<=i<=N, 0<=j<=V

1. 不装入第i件物品，即`dp[i−1][j]`；
2. 装入第i件物品（前提是能装下），即`dp[i][j−vi[i]]+w[i]`

由于每种物品有无限个，即装入第i种商品后还可以再继续装入第种商品，故转移到`dp[i][j−vi[i]]+w[i]`

状态转移方程：`dp[i][j] = max(dp[i−1][j], dp[i][j−v[i]]+w[i]) // j >= v[i]`

基础代码如下

```cpp
void pack_full(){
    vector<vector<int>>dp(N+1, vector<int>(V+1, 0));
    for(int i=1; i<=N; i++){
        for(int j=1; j<=V; j++){
            if(j >= vi[i])
                dp[i][j] = max(dp[i-1][j], dp[i][j-vi[i]] + wi[i]);
            else
                dp[i][j] = dp[i-1][j];
        }
    }
    cout << dp[N][V];
}
```

由于dp[i][j]只和上一轮的数据dp[i-1][j]相关，并不需要保留更前的数据，所以可以只用一维dp数组。j的遍历改为从小到大，可以防止j值较小时被覆盖（新的dp[j]需要使用旧的dp[j-1]计算）

因此可以化简为：

```cpp
void pack_full(){
    vector<int>dp(V+1);
    for(int i=1; i<=N; i++){
        for(int j=vi[i]; j<=V; j++)
            dp[j] = max(dp[j], dp[j-vi[i]]+wi[i]);
    }
    cout << dp[V];
}
```

上述两个代码的数据初始化如下

```cpp
vector<int> vi(N+1, 0);
vector<int> wi(N+1, 0);
vi[0] = 0;
wi[0] = 0;
for(int i=1; i<=N; i++)
    cin >> vi[i] >> wi[i];
```

更加优化：**转换成01背包**

简单的思想——考虑到第 i 种物品最多装入 V/vi[i] 件，于是可以把第 i 种物品转化为V/vi[i] 件体积及价值均不变的物品，然后求解这个01背包问题。（如物体i变成si[i]个体积vi[i]，价值wi[i]的物体）

更高效的转化方法是采用二进制的思想——把第 i 种物品拆成体积为 $v_i[i]2^k$，价值 $w_i[i]2^k$的若干个物体，其中 k 取遍满足 $sum( 2^k) ≤ V / v_i[i]$的非负整数。这是因为不管最优策略选几件第 i 种物品，总可以表示成若干个刚才这些物品的和（例：13 = 1 + 4 + 8）。这样就将转换后的物品数目降成了对数级别。

```cpp
void pack_full(){
    vector<int> dp(V+1, 0);
    vector<int> vi;
    vector<int> wi;
    for(int i=1; i<=N; i++){
        int v, w, tmp, k;
        cin >> v >> w;

        tmp = V/v; // 背包容量的限制
        for(k=1; tmp>0; k<<=1){
            int amount = min(k, tmp);
            vi.push_back(k*v);
            wi.push_back(k*w);
            tmp -= k;
        }
    }

    for (int i = 0; i < vi.size(); ++i) {
        for (int j = V; j >= vi[i]; --j) {
            dp[j] = max(dp[j], dp[j - vi[i]] + wi[i]);
        }
    }
    cout << dp[V];
}
```

### 多重背包问题

dp[i][j]表示将前i种物品装进限重为j的背包可以获得的最大价值,  0<=i<=N, 0<=j<=V

此时的分析和完全背包差不多，也是从装入第 i 种物品多少件出发：装入第i种物品0件、1件、...n[i]件（还要满足不超过限重）。所以状态方程为：

1. 不装入第i件物品，即`dp[i−1][j]`；
2. 装入第i件物品（前提是能装下），即  `k为装入第i种物品的件数, k <= min(n[i], j/w[i])`
`dp[i][j] = max{(dp[i-1][j − k*w[i]] + k*v[i]) for every k}`

基础代码如下

```cpp
void pack_multi(){
    vector<vector<int>>dp(N+1, vector<int>(V+1, 0));
    for(int i=1; i<=N; i++){
        for(int j=1; j<=V; j++){
            int tmp = 0;
            // dp[i][j] = max(dp[i-1][j-k*vi[i]] + k*wi[i]);
            for(int k=0; k<=min(si[i], j/vi[i]); k++) // 数量限制和容量限制中取最小
                if(tmp < dp[i-1][j-k*vi[i]] + k*wi[i])
                    tmp = dp[i-1][j-k*vi[i]] + k*wi[i];
            dp[i][j] = tmp;
        }
    }
    cout << dp[N][V];
}
```

由于dp[i][j]只和上一轮的数据dp[i-1][j]相关，并不需要保留更前的数据，所以可以只用一维dp数组。j的遍历改为从大到小，可以防止j值较小时被覆盖（新的dp[j]需要使用旧的dp[j-1]计算）

因此可以化简为：

```cpp
void pack_multi(){
    vector<int>dp(V+1);
    for(int i=1; i<=N; i++){
        for(int j=V; j>=vi[i]; j--){
            for(int k=0; k<=min(si[i], j/vi[i]); k++)
                dp[j] = max(dp[j], dp[j-k*vi[i]] + k*wi[i]);
        }
    }
    cout << dp[V];
}
```

上述两个代码的数据初始化如下

```cpp
  vector<int> vi(N+1, 0);
    vector<int> wi(N+1, 0);
    vi[0] = 0;
    wi[0] = 0;
    for(int i=1; i<=N; i++)
        cin >> vi[i] >> wi[i];
```

更加优化：**转换成01背包**

更高效的转化方法是采用二进制的思想——把第 i 种物品拆成体积为 $v_i[i]2^k$，价值 $w_i[i]2^k$的若干个物体，其中 k 取遍满足 $sum(2^k) ≤ s_i[i]$ 的非负整数。这是因为不管最优策略选几件第 i 种物品，总可以表示成若干个刚才这些物品的和（例：13 = 1 + 4 + 8）。这样就将转换后的物品数目降成了对数级别。

```cpp
void pack_multi(){
    vector<int>dp(V+1, 0);
    vector<int> vi;
    vector<int> wi;
    for(int i=1; i<=N; i++){
        int v, w, s, k;
        cin >> v >> w >> s;
        for (int k = 1; s > 0; k <<= 1) { // k <<= 1 相当于 k=k*2
            int amount = min(k, s);
            vi.push_back(amount*v);
            wi.push_back(amount*w);
            s -= amount;
        }
    }

    for (int i = 0; i < vi.size(); ++i) {
        for (int j = V; j >= vi[i]; --j)
            dp[j] = max(dp[j], dp[j - vi[i]] + wi[i]);
    }
    cout << dp[V];
}
```

## 股票购买问题

最多进行max_k次交易，要求收益最大

```cpp
// 后一天只依赖于前一天的情况
int maxProfit(vector<int>& prices, int max_k) {
  int size = prices.size();
  if (size < 2) return 0;
  vector<vector<int>> dp(max_k + 1, vector<int>(2, 0));
  for (int k = 1; k <= max_k; ++k) {
    dp[k][0] = 0;
    dp[k][1] = -prices[0];
  }
  for (int i = 1; i < size; ++i) {
    for (int k = max_k; k >=1; --k) {
      dp[k][0] = max(dp[k][0], dp[k][1] + prices[i]);
      dp[k][1] = max(dp[k][1], dp[k - 1][0] - prices[i]);
    }
  }
  return dp[max_k][0];
}
```

## 其他

现需要将一根长为正整数 bamboo_len 的竹子砍为若干段，每段长度均为正整数。请返回每段竹子长度的最大乘积是多少。

输入: bamboo_len = 12, 输出: 81

```cpp
if (n == 2) return 1;
if (n == 3) return 2;
vector<int> dp(n + 1);
dp[1] = 1;
dp[2] = 2;
dp[3] = 3;
for (int i = 4; i <=n ; ++i) {
  int maxVal = 0;
  // 对半拆
  for (int j = 1; j <= i / 2; ++j) {
    maxVal = max(maxVal, dp[j] * dp[i - j]);
  }
  dp[i] = maxVal;
}
return dp[n];
```

# 位运算

1.实现了判断 n 的二进制表示中有多少个 1

```cpp
while (n != 0) {
    count += n & 1; // 与二进制最低位 “与”
    n = n >> 1;    // n (逻辑)右移
}
```

2.用 `^` 和 `&` 模拟加法(a + b)

```cpp
while (b != 0) {
  int tmp = a ^ b; // 1 ^ 1 = 0, 1 ^ 0 = 1
  b = (a & b) << 1; // 作为进位的部分
  a = tmp;
}
```

`a ^ a = 0`, `0 ^ a = a` -> `^` 运算用来判断数相等

`a & b` 是按位与，如果结果大于等于1

3.LCR 177. 撞色搭配

整数数组 sockets 记录了一个袜子礼盒的颜色分布情况，其中 sockets[i] 表示该袜子的颜色编号。礼盒中除了一款撞色搭配的袜子，每种颜色的袜子均有两只。请设计一个程序，在时间复杂度 O(n)，空间复杂度O(1) 内找到这双撞色搭配袜子的两个颜色编号。

```cpp
vector<int> sockCollocation(vector<int>& sockets) {
  int size = sockets.size();
  if (size == 2) return sockets;
  int ret = 0;
  for (int i : sockets)
    ret ^= i; // 全部异或起来
  int idx = 0;
  // 找到 ret 中第一个不为 0 的位置来做分割
  while (!(ret & 1)) {
    ++idx;
    ret = ret >> 1;
  }
  int diff = 1 << idx;
  // 分为两组，再做一遍
  int res1 = 0, res2 = 0;
  for (int i : sockets) {
    if (i & diff) res1 ^= i;
    else res2 ^= i;
  }
  return {res1, res2};
}
```

# 模拟法 / 设计

1.leetcode43 字符串相乘

```cpp
string multiply(string num1, string num2) {
    if (num1.size() == 0 || num2.size() == 0) return "";
    if (num1 == "0" || num2 == "0") return "0";
    vector<int> res(num1.size() + num2.size(), 0);
    int offset = 0;
    for (auto iter1 = num1.rbegin(); iter1 != num1.rend(); ++iter1) {
        int val1 = *iter1 - '0';
        int time1 = iter1 - num1.rbegin();
        for (auto iter2 = num2.rbegin(); iter2 != num2.rend(); ++iter2) {
            int val2 = *iter2 - '0';
            auto time2 = iter2 - num2.rbegin();
            int now = val1 * val2 + res[time1 + time2] + offset;
            offset = now / 10;
            res[time1 + time2] = now % 10;
        }
        res[time1 + num2.size()] = offset;
        offset = 0;
    }
    string result;
    auto iter = res.rbegin();
    auto end = res.rend();
    while (iter != end && *iter == 0) ++iter;
    while (iter != end) {
        result += std::to_string(*iter);
        ++iter;
    }
    return result;
}
```

2.Leetcode146 实现LRU

```cpp
class LRUCache {
public:
  LRUCache(int capacity) : capacity(capacity) {}
  int get(int key) {
    if (!cache.contains(key)) return -1;
    // 把被访问的元素提前
    auto iter = cache[key];
    int res = (*iter).second;
    values.push_front(*iter);
    // 更新
    values.erase(iter);
    cache[key] = values.begin();
    return res;
  }
  void put(int key, int value) {
    if (cache.contains(key)) {
      values.erase(cache[key]);
    } else if (cache.size() == capacity) {
      // 删掉该废弃的
      cache.erase(values.back().first);
      values.pop_back();
    }
    values.push_front(std::make_pair(key, value));
    cache[key] = values.begin();
  }

private:
  const int capacity;
  unordered_map<int, list<std::pair<int, int>>::iterator> cache; // key, value的iter
  list<std::pair<int, int>> values;
}
```

3.LeetCode208. 实现 Trie (前缀树)

```cpp
class TrieNode {
public:
  bool isWord;
  vector<TrieNode*> childs; // 26个字母
  TrieNode() : isWord(false), childs(26, nullptr) {}
};
class Trie {
public:
    Trie() {root = new TrieNode;}

    void insert(string word) {
      int size = word.size();
      if (size == 0) return;
      auto curr = root;
      for (int i = 0; i < size; ++i) {
        int idx = word[i] - 'a';
        if (curr->childs[idx] == nullptr) {
          curr->childs[idx] = new TrieNode;
        }
        curr = curr->childs[idx];
      }
      curr->isWord = true;
    }

    bool search(string word) {
      int size = word.size();
      if (size == 0) return true;
      auto curr = root;
      for (int i = 0; i < size; ++i) {
        int idx = word[i] - 'a';
        if (curr->childs[idx] == nullptr) return false;
        curr = curr->childs[idx];
      }
      return curr->isWord;
    }

    bool startsWith(string prefix) {
      int size = prefix.size();
      if (size == 0) return true;
      auto curr = root;
      for (int i = 0; i < size; ++i) {
        int idx = prefix[i] - 'a';
        if (curr->childs[idx] == nullptr) return false;
        curr = curr->childs[idx];
      }
      return true;
    }
private:
    TrieNode *root;
};
```

4.LeetCode211. 添加与搜索单词，搜索时 `.` 可以表示任意一个字符

```cpp
struct TrieNode {
  bool isWord;
  unordered_map<char, TrieNode*> childs;
  TrieNode() : isWord(false) {}
};
class WordDictionary {
private:
  TrieNode *root;
public:
  WordDictionary() {
    root = new TrieNode();
  }

  void addWord(string word) {
    TrieNode *curr = root;
    for (char c : word) {
      if (!(curr->childs).contains(c)) {
        (curr->childs)[c] = new TrieNode;
      }
      curr = (curr->childs)[c];
    }
    curr->isWord = true;
  }

  bool search_new(string &word, int idx, TrieNode *curr) {
    int size = word.size();
    if (idx == size) return curr->isWord;
    char c = word[idx];
    if (c == '.') {
      for (auto [cc, child] : curr->childs) {
        if (search_new(word, idx + 1, child))
          return true;
      }
      return false;
    }
    if ((curr->childs).contains(c))
      return search_new(word, idx + 1, (curr->childs)[c]);
    return false;
  }
  bool search(string word) {
    TrieNode *curr = root;
    return search_new(word, 0, curr);
  }
};
```
