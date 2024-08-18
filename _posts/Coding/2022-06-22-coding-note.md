---
title: 算法刷题笔记
author: tfruan
date: 2022-06-22 20:00:00 +0800
categories: [Coding]
tags: [Coding, Algorithm]
---

[2024/08/10 update]再不刷题，秋招咋办！！

# 双指针

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

# 栈与队列

## 单调栈

1. LCR 183. 望远镜中最高的海拔

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

# 排序

## 快排

每次挑出一个基准，比基准小的放左边，比基准大的放右边

```cpp
template<typename T>
void quick_sort(T arr[], int start, int end){
    if(start >= end)return;
    int l = start - 1;
    int r = end + 1;
    T pvoit = arr[l+(r-l)/2];
    **while(l < r){
        do l++; while(arr[l] < pvoit);
        do r--; while(arr[r] > pvoit);
        if( l < r) swap(arr[l], arr[r]);
    }**
    quick_sort(arr, start, r);
    quick_sort(arr, r+1, end);
}
```

![sort](/assets/img/blog/img_coding_note/computational_complexity.png)

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

# 搜索与回溯

## 深度优先 DFS

1. LCR 130. 衣橱整理

```cpp
int dfs(vector<vector<bool>> &visited, int i, int j, int cnt, int m, int n) {
    auto digit = [](int k) -> int{
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

## 广度优化 BFS

在树上表现为层序遍历

## 层序遍历

1. LCR 149. 彩灯装饰记录 I

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

# 分治

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
  if (l == r + 1)
    return trave(postorder, l - 1, start) && trave(postorder, root - 1, r + 1);
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

# 树

## 层序遍历

1. LCR 149. 彩灯装饰记录 I

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

## 递归

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

# 动态规划问题

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

更更加优化：单调队列
