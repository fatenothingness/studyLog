import sun.jvm.hotspot.debugger.Page;

import java.util.*;

public class SwordToOffer {
    /**
     * 剑指 Offer 09. 用两个栈实现队列
     * 用两个栈实现一个队列。
     * 队列的声明如下，请实现它的两个函数 appendTail 和 deleteHead ，分别完成在队列尾部插入整数和在队列头部删除整数的功能。
     * (若队列中没有元素，deleteHead 操作返回 -1 )
     */
    class CQueue {

        public Stack<Integer> stackA;

        public Stack<Integer> stackB;

        public CQueue() {
            this.stackA = new Stack<>();
            this.stackB = new Stack<>();
        }

        public void appendTail(int value) {
            if (!stackB.isEmpty()) {
                while (!stackB.isEmpty()) {
                    stackA.push(stackB.pop());
                }
            }
            stackA.push(value);
        }

        public int deleteHead() {
            if (stackA.isEmpty() && stackB.isEmpty()) {
                return -1;
            } else {
                while (!stackA.isEmpty()) {
                    stackB.push(stackA.pop());
                }
                return stackB.pop();
            }
        }
    }


    /**
     * 剑指 Offer 10- I. 斐波那契数列
     * 写一个函数，输入 n ，求斐波那契（Fibonacci）数列的第 n 项（即 F(N)）。斐波那契数列的定义如下：
     * <p>
     * F(0) = 0,   F(1) = 1
     * F(N) = F(N - 1) + F(N - 2), 其中 N > 1.
     * 斐波那契数列由 0 和 1 开始，之后的斐波那契数就是由之前的两数相加而得出。
     * <p>
     * 答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。
     */
    public int fib(int n) {
        if (n < 2) {
            return n;
        }
        int[] dp = new int[n + 1];
        dp[0] = 0;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            dp[i] = (dp[i - 1] + dp[i - 2]) % 1000000007;
        }
        return dp[n];
    }

    /**
     * 剑指 Offer 10- II. 青蛙跳台阶问题
     * 一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个 n 级的台阶总共有多少种跳法。
     * 答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。
     */
    public int numWays(int n) {
        if (n == 0) {
            return 1;
        }
        if (n == 1) {
            return 1;
        }
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            dp[i] = (dp[i - 1] + dp[i - 2]) % 1000000007;
        }
        return dp[n];
    }

    /**
     * 剑指 Offer 11. 旋转数组的最小数字
     * 把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
     * 输入一个递增排序的数组的一个旋转，输出旋转数组的最小元素。例如，数组 [3,4,5,1,2] 为 [1,2,3,4,5] 的一个旋转，该数组的最小值为1。
     */
    public int minArray(int[] numbers) {
        for (int i = 0; i < numbers.length - 1; i++) {
            if (numbers[i] > numbers[i + 1]) {
                return numbers[i + 1];
            }
        }
        return numbers[0];
    }

    /**
     * 剑指 Offer 12. 矩阵中的路径
     * 给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。
     * 单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。
     * 例如，在下面的 3×4 的矩阵中包含单词 "ABCCED"（单词中的字母已标出）。
     */
    public boolean exist(char[][] board, String word) {
        int m = board.length;
        int n = board[0].length;
        boolean[][] tmp = new boolean[m][n];
        char[] s = word.toCharArray();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (dfs(board, s, i, j, 0)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean dfs(char[][] board, char[] s, int i, int j, int k) {
        if (i >= board.length || i < 0 || j < 0 || j >= board[0].length || s[k] != board[i][j]) {
            return false;
        }
        if (k == s.length - 1) {
            return true;
        }
        board[i][j] = '\0';
        boolean res = dfs(board, s, i - 1, j, k + 1) || dfs(board, s, i + 1, j, k + 1) || dfs(board, s, i, j - 1, k + 1) || dfs(board, s, i, j + 1, k + 1);
        board[i][j] = s[k];
        return res;
    }

    /**
     * 剑指 Offer 38. 字符串的排列
     * 输入一个字符串，打印出该字符串中字符的所有排列。
     * 你可以以任意顺序返回这个字符串数组，但里面不能有重复元素。
     */
    public String[] permutation(String s) {
        char[] c = s.toCharArray();
        Set<String> set = new HashSet<>();
        StringBuilder sb = new StringBuilder();
        dps(c, set, sb);
        String[] res = set.toArray(new String[set.size()]);
        return res;
    }

    private void dps(char[] c, Set<String> set, StringBuilder sb) {
        if (sb.length() == c.length) {
            set.add(sb.toString());
            return;
        }
        for (int i = 0; i < c.length; i++) {
            if (c[i] != '-') {
                char t = c[i];
                sb.append(t);
                c[i] = '-';
                dps(c, set, sb);
                c[i] = t;
                sb.deleteCharAt(sb.length() - 1);
            }
        }
    }

    /**
     * 剑指 Offer 03. 数组中重复的数字
     * 找出数组中重复的数字。
     * 在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。
     * 数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。
     * 输入：
     * [2, 3, 1, 0, 2, 5, 3]
     * 输出：2 或 3
     */
    public int findRepeatNumber(int[] nums) {
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(nums[i])) {
                return nums[i];
            } else {
                map.put(nums[i], 1);
            }
        }
        return -1;
    }

    /**
     * 剑指 Offer 04. 二维数组中的查找
     * 在一个 n * m 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。
     * 请完成一个高效的函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
     */
    public boolean findNumberIn2DArray(int[][] matrix, int target) {
        int n = matrix.length;
        if(n==0){
            return false;
        }
        int m = matrix[0].length;
        if(m==0){
            return false;
        }
        for (int i = 0; i < n; i++) {
            int l = 0;
            int r = m-1;
            while (l < r) {
                int mid = (l + r + 1) >> 1;
                if (matrix[i][mid] > target) {
                    r = mid - 1;
                } else if (matrix[i][mid] < target) {
                    l = mid;
                } else {
                    return true;
                }
            }
            if(matrix[i][l]==target){
                return true;
            }
        }
        return false;
    }

    /**
     * 剑指 Offer 15. 二进制中1的个数
     * 请实现一个函数，输入一个整数（以二进制串形式），输出该数二进制表示中 1 的个数。
     * 例如，把 9 表示成二进制是 1001，有 2 位是 1。因此，如果输入 9，则该函数输出 2。
     */
    public int hammingWeight(int n) {
        int res = 0;
        while(n!=0){
            n&=n-1;
            res++;
        }
        return res;
    }

    /**
     * 剑指 Offer 06. 从尾到头打印链表
     * 输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。
     */
    public int[] reversePrint(ListNode head) {
        Stack<Integer> stack = new Stack<>();
        while(head!=null){
            stack.push(head.val);
            head=head.next;
        }
        int[] res = new int[stack.size()];
        for(int i=0;i<res.length;i++){
            res[i] = stack.pop();
        }
        return res;
    }

    /**
     * 剑指 Offer 05. 替换空格
     * 请实现一个函数，把字符串 s 中的每个空格替换成"%20"。
     */
    public String replaceSpace(String s) {
        StringBuilder sb = new StringBuilder();
        for(int i=0;i<s.length();i++){
            if(s.charAt(i)==' '){
                sb.append("%20");
            }else {
                sb.append(s.charAt(i));
            }
        }
        return sb.toString();
    }

    /**
     * 剑指 Offer 07. 重建二叉树
     * 输入某二叉树的前序遍历和中序遍历的结果，请重建该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
     */
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        //哈希表保存中序遍历的结果
        HashMap<Integer,Integer> map = new HashMap<>();
        for(int i=0;i<inorder.length;i++){
            map.put(inorder[i],i);
        }
        return process(0,0,preorder.length-1,preorder,map);
    }
    // 参数含义：root为前序遍历中根节点的位置，left为中序遍历中的左边界，rigth为右边界
    private TreeNode process(int root, int left, int right,int[] preorder,HashMap<Integer,Integer> map){
        if(left>right){
            return null;
        }
        TreeNode res = new TreeNode(preorder[root]);
        //获取当前节点在中序遍历的位置，前半部为左子树，后半部为右子树
        Integer i = map.get(preorder[root]);
        //在前序遍历的左树区间
        res.left = process(root+1,left,i-1,preorder,map);
        //i-left 为左树区间的长度，在前序遍历的，右树的根节点为 当前根节点位置+左树长度+1
        res.right = process(root + (i - left) + 1,i+1,right,preorder,map);
        return res;
    }


    /**
     * 剑指 Offer 13. 机器人的运动范围
     * 地上有一个m行n列的方格，从坐标 [0,0] 到坐标 [m-1,n-1] 。
     * 一个机器人从坐标 [0, 0] 的格子开始移动，它每次可以向左、右、上、下移动一格（不能移动到方格外），也不能进入行坐标和列坐标的数位之和大于k的格子。
     * 例如，当k为18时，机器人能够进入方格 [35, 37] ，因为3+5+3+7=18。但它不能进入方格 [35, 38]，因为3+5+3+8=19。请问该机器人能够到达多少个格子？
     */
    public int movingCount(int m, int n, int k) {
        if (k == 0) {
            return 1;
        }
        //记录可能的数位和
        int max = Math.max(m,n);
        int[] map = new int[max];
        for(int i =0;i<max;i++){
            map[i] = getSum(i);
        }
        //记录每个位置是否可达
        boolean[][] vis = new boolean[m][n];
        int ans = 1;
        vis[0][0] = true;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if ((i == 0 && j == 0) || map[i] + map[j] > k) {
                    continue;
                }
                // 边界判断，如果第一行则不判断上面的可行性，如果在第一列则不判断左边的可行性，其余只要满足左边或上面可达，这该点也可达。
                if(i==0){
                    vis[i][j] = vis[i][j-1];
                } else if (j == 0) {
                    vis[i][j] = vis[i-1][j];
                }else {
                    vis[i][j] = vis[i][j-1]|vis[i-1][j];
                }
                ans += vis[i][j] ? 1 : 0;
            }
        }
        return ans;
    }

    private Integer getSum(int a){
        int res = 0;
        while(a!=0){
            res +=a%10;
            a/=10;
        }
        return  res;
    }

    /**
     * 剑指 Offer 14- I. 剪绳子
     * 给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 k[0],k[1]...k[m-1] 。
     * 请问 k[0]*k[1]*...*k[m-1] 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。
     */
    public int cuttingRope(int n) {
        if (n <= 3) return n - 1;
        int[] dp = new int[n+1];
        dp[0] = 0;
        dp[1] = 1;
        dp[2] = 2;
        dp[3] = 3;
        /*
         *  外层循环i表示每一段要剪的绳子，去掉特殊情况从4开始
         *  内层循环j表示将绳子剪成长度为j和i-j的两段
         *  这样双层循环就相当于从下向上完成了剪绳子的逆过程
         *  （剪绳子本来是将大段的绳子剪成小段，然后再在每小段上继续剪）
         *  双层循环中外层循环从4开始一直到原始绳子长度n，每一段都到内层循环进行剪绳子
         *  这样就得到长度在[4, n]区间内的每段绳子剪过之后的最大乘积
         *  dp[i]记录当前长度绳子剪过之后的最大乘积
         */
        for (int i = 4; i <=n; i++) {
            for (int j = 1; j < i; j++) {
                dp[i] = Math.max(dp[i], dp[j] * dp[i - j]);
            }
        }
        /* 返回剪绳子的最大乘积 */
        return dp[n];
    }

    /**
     * 剑指 Offer 16. 数值的整数次方
     * 实现 pow(x, n) ，即计算 x 的 n 次幂函数（即，xn）。不得使用库函数，同时不需要考虑大数问题。
     */
    public double myPow(double x, int n) {
        //快速幂方法
        if(x == 0) return 0;
        long b = n;
        double res = 1.0;
        if(b < 0) {
            x = 1 / x;
            b = -b;
        }
        while(b > 0) {
            if((b & 1) == 1) res *= x;
            x *= x;
            b >>= 1;
        }
        return res;
    }

    /**
     * 剑指 Offer 17. 打印从1到最大的n位数
     * 输入数字 n，按顺序打印出从 1 到最大的 n 位十进制数。比如输入 3，则打印出 1、2、3 一直到最大的 3 位数 999。
     */
    public int[] printNumbers(int n) {
        int num = 1;
        for(int i=0;i<n;i++){
            num *=10;
        }
        num--;
        int[] res = new int[num];
        for(int i=0;i<num;i++){
            res[i] = i+1;
        }
        return res;
    }

    /**
     * 剑指 Offer 18. 删除链表的节点
     * 给定单向链表的头指针和一个要删除的节点的值，定义一个函数删除该节点。
     * 返回删除后的链表的头节点。
     */
    public ListNode deleteNode(ListNode head, int val) {
        ListNode tmp = new ListNode(0,head);
        ListNode root = head;
        if(root.val==val){
            return root.next;
        }
        while(root!=null){
            if(root.val==val){
                tmp.next = root.next;
                break;
            }else {
                root=root.next;
                tmp=tmp.next;
            }
        }
        return head;
    }

    /**
     * 剑指 Offer 19. 正则表达式匹配
     * 请实现一个函数用来匹配包含'. '和'*'的正则表达式。
     * 模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（含0次）。
     * 在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但与"aa.a"和"ab*a"均不匹配。
     */
    public boolean isMatch(String A, String B) {
        int n = A.length();
        int m = B.length();
        boolean[][] f = new boolean[n + 1][m + 1];

        for (int i = 0; i <= n; i++) {
            for (int j = 0; j <= m; j++) {
                //分成空正则和非空正则两种
                if (j == 0) {
                    f[i][j] = i == 0;
                } else {
                    //非空正则分为两种情况 * 和 非*
                    if (B.charAt(j - 1) != '*') {
                        if (i > 0 && (A.charAt(i - 1) == B.charAt(j - 1) || B.charAt(j - 1) == '.')) {
                            f[i][j] = f[i - 1][j - 1];
                        }
                    } else {
                        //碰到 * 了，分为看和不看两种情况
                        //不看
                        if (j >= 2) {
                            f[i][j] |= f[i][j - 2];
                        }
                        //看
                        if (i >= 1 && j >= 2 && (A.charAt(i - 1) == B.charAt(j - 2) || B.charAt(j - 2) == '.')) {
                            f[i][j] |= f[i - 1][j];
                        }
                    }
                }
            }
        }
        return f[n][m];
    }

    /**
     * 剑指 Offer 21. 调整数组顺序使奇数位于偶数前面
     * 输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有奇数位于数组的前半部分，所有偶数位于数组的后半部分。
     */
    public int[] exchange(int[] nums) {
        int head = 0;
        int end = nums.length-1;
        while(head<end){
            if(nums[head]%2==0){
                swap(nums,head,end);
                end--;
            }else {
                head++;
            }
        }
        return nums;
    }

    private void swap(int[] n ,int a,int b){
        int tmp = n[a];
        n[a] = n[b];
        n[b] = tmp;
    }

    /**
     * 剑指 Offer 22. 链表中倒数第k个节点
     * 输入一个链表，输出该链表中倒数第k个节点。为了符合大多数人的习惯，本题从1开始计数，即链表的尾节点是倒数第1个节点。
     * 例如，一个链表有 6 个节点，从头节点开始，它们的值依次是 1、2、3、4、5、6。这个链表的倒数第 3 个节点是值为 4 的节点
     */
    public ListNode getKthFromEnd(ListNode head, int k) {
        ListNode f = head;
        ListNode res = head;
        while(k>0){
            f=f.next;
            k--;
        }
        while(f!=null){
            f=f.next;
            res=res.next;
        }
        return res;
    }

    /**
     * 剑指 Offer 24. 反转链表
     * 定义一个函数，输入一个链表的头节点，反转该链表并输出反转后链表的头节点。
     */
    public ListNode reverseList(ListNode head) {
        return process(head);
    }

    private ListNode process(ListNode head){
        if (head == null || head.next == null) {
            return head;
        }
        ListNode newHead = process(head.next);
        head.next.next = head;
        head.next = null;
        return newHead;
    }

    /**
     * 剑指 Offer 25. 合并两个排序的链表
     * 输入两个递增排序的链表，合并这两个链表并使新链表中的节点仍然是递增排序的。
     */
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode res = new ListNode(1);
        ListNode ans = res;
        while(l1!=null&&l2!=null){
            if(l1.val<=l2.val){
                res.next=l1;
                l1=l1.next;
            }else {
                res.next=l2;
                l2=l2.next;
            }
            res=res.next;
        }
        if(l1!=null){
            res.next = l1;
        }else {
            res.next = l2;
        }
        return ans.next;
    }

    /**
     * 剑指 Offer 26. 树的子结构
     * 输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)
     * B是A的子结构， 即 A中有出现和B相同的结构和节点值。
     */
    public boolean isSubStructure(TreeNode A, TreeNode B) {
        if(B==null){
            return false;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(A);
        while(!queue.isEmpty()){
            TreeNode now = queue.poll();
            if(now.val== B.val){
                if(process(now,B)){
                    return true;
                }
            }
            if(now.left!=null) queue.add(now.left);
            if(now.right!=null) queue.add(now.right);
        }
        return false;
    }

    public boolean process(TreeNode A, TreeNode B){
        if(B==null){
            return true;
        }
        if(A==null||A.val!=B.val){
            return false;
        }
        return process(A.left,B.left)&&process(A.right,B.right);
    }
}

