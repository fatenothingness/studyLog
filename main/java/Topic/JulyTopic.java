package Topic;

import util.TreeNode;

import java.util.*;


public class JulyTopic {
    public static final int mod = (int)1e9+7;

    /**
     * 1818. 绝对差值和
     * 给你两个正整数数组 nums1 和 nums2 ，数组的长度都是 n 。
     *
     * 数组 nums1 和 nums2 的 绝对差值和 定义为所有 |nums1[i] - nums2[i]|（0 <= i < n）的 总和（下标从 0 开始）。
     *
     * 你可以选用 nums1 中的 任意一个 元素来替换 nums1 中的 至多 一个元素，以 最小化 绝对差值和。
     *
     * 在替换数组 nums1 中最多一个元素 之后 ，返回最小绝对差值和。因为答案可能很大，所以需要对 109 + 7 取余 后返回。
     */
    public int minAbsoluteSumDiff(int[] nums1, int[] nums2) {
        int n = nums1.length;
        int[] sorted = nums1.clone();
        Arrays.sort(sorted);
        long sum = 0, max = 0;
        for (int i = 0; i < n; i++) {
            int a = nums1[i], b = nums2[i];
            if (a == b) continue;
            int x = Math.abs(a - b);
            sum += x;
            int l = 0, r = n - 1;
            while (l < r) {
                int mid = l + r + 1 >> 1;
                if (sorted[mid] <= b) l = mid;
                else r = mid - 1;
            }
            int nd = Math.abs(sorted[r] - b);
            if (r + 1 < n) nd = Math.min(nd, Math.abs(sorted[r + 1] - b));
            if (nd < x) max = Math.max(max, x - nd);
        }
        return (int)((sum - max) % mod);
    }

    /**
     * 1846. 减小和重新排列数组后的最大元素
     * 给你一个正整数数组 arr 。请你对 arr 执行一些操作（也可以不进行任何操作），使得数组满足以下条件：
     * arr 中 第一个 元素必须为 1 。
     * 任意相邻两个元素的差的绝对值 小于等于 1 ，也就是说，对于任意的 1 <= i < arr.length （数组下标从 0 开始），都满足 abs(arr[i] - arr[i - 1]) <= 1 。abs(x) 为 x 的绝对值。
     */
    public int maximumElementAfterDecrementingAndRearranging(int[] arr) {
        Arrays.sort(arr);
        int n = arr.length;
        arr[0] = 1;
        for(int i=1;i<n;i++){
            if(arr[i]-arr[i-1]>1){
                arr[i]=arr[i-1]+1;
            }
        }
        return arr[n-1];
    }


    /**
     * 剑指 Offer 53 - I. 在排序数组中查找数字 I
     * 统计一个数字在排序数组中出现的次数。
     */
    public int search(int[] nums, int target) {
        int l = 0;
        int r = nums.length-1;
        while(l<r){
            int mid = l+(r-l>>1);
            if(nums[mid]>=target){
                r=mid;
            }else {
                l = mid+1;
            }
        }
        int res = 0;
        for(int i=l;i<nums.length;i++){
            if(nums[i]==target){
                res++;
            }else {
                break;
            }
        }
        return res;
    }

    /**
     * 1838. 最高频元素的频数
     * 元素的 频数 是该元素在一个数组中出现的次数。
     *
     * 给你一个整数数组 nums 和一个整数 k 。在一步操作中，你可以选择 nums 的一个下标，并将该下标对应元素的值增加 1 。
     *
     * 执行最多 k 次操作后，返回数组中最高频元素的 最大可能频数 。
     *
     *
     */
    public int maxFrequency(int[] nums, int k) {
        int n = nums.length;
        Arrays.sort(nums);
        int res = 1;
        int left = n-2;
        int right  = n-1;
        while(left>=0){
            if(nums[right]==nums[left]||nums[right]-nums[left]<=k){
                res = Math.max(res,right-left+1);
                k-=nums[right]-nums[left];
                left--;
            }else {
                k+=(nums[right]-nums[right-1])*(right-left-1);
                right--;
            }
        }
        return res;
    }

    /**
     * 1877. 数组中最大数对和的最小值
     * 一个数对 (a,b) 的 数对和 等于 a + b 。最大数对和 是一个数对数组中最大的 数对和 。
     *
     * 比方说，如果我们有数对 (1,5) ，(2,3) 和 (4,4)，最大数对和 为 max(1+5, 2+3, 4+4) = max(6, 5, 8) = 8 。
     * 给你一个长度为 偶数 n 的数组 nums ，请你将 nums 中的元素分成 n / 2 个数对，使得：
     *
     * nums 中每个元素 恰好 在 一个 数对中，且
     * 最大数对和 的值 最小 。
     * 请你在最优数对划分的方案下，返回最小的 最大数对和 。
     */

    public int minPairSum(int[] nums) {
        int n = nums.length;
        Arrays.sort(nums);
        int res = 0;
        for(int i=0;i<n;i++){
            res = Math.max(nums[i]+nums[n-i-1],res);
        }
        return res;
    }

    class Node {
        int val;
        Node next;
        Node random;

        public Node(int val) {
            this.val = val;
            this.next = null;
            this.random = null;
        }
    }

    /**
     * 138. 复制带随机指针的链表
     * 给你一个长度为 n 的链表，每个节点包含一个额外增加的随机指针 random ，该指针可以指向链表中的任何节点或空节点。
     * 构造这个链表的 深拷贝。 深拷贝应该正好由 n 个 全新 节点组成，其中每个新节点的值都设为其对应的原节点的值。新节点的 next 指针和 random 指针也都应指向复制链表中的新节点，并使原链表和复制链表中的这些指针能够表示相同的链表状态。复制链表中的指针都不应指向原链表中的节点 。
     */
    public Node copyRandomList(Node head) {
        if(head==null){
            return null;
        }
        HashMap<Node,Integer> map =new HashMap<>();
        List<Node> list =new ArrayList<>();
        Node t = head;
        int i=0;
        while(t!=null){
            map.put(t,i++);
            Node node = new Node(t.val);
            list.add(node);
            t=t.next;
        }
        for(int j=0;j<list.size()-1;j++){
            Node now = list.get(j);
            Node next = list.get(j+1);
            now.next=next;
        }
        int j=0;
        while(head!=null){
            if(head.random==null){
                list.get(j++).random=null;
            }else {
                Node random = list.get(map.get(head.random));
                list.get(j++).random=random;
            }
            head=head.next;
        }
        return list.get(0);
    }

    /**
     * 1893. 检查是否区域内所有整数都被覆盖
     * 给你一个二维整数数组 ranges 和两个整数 left 和 right 。每个 ranges[i] = [starti, endi] 表示一个从 starti 到 endi 的 闭区间 。
     *
     * 如果闭区间 [left, right] 内每个整数都被 ranges 中 至少一个 区间覆盖，那么请你返回 true ，否则返回 false 。
     *
     * 已知区间 ranges[i] = [starti, endi] ，如果整数 x 满足 starti <= x <= endi ，那么我们称整数x 被覆盖了。
     */
    public boolean isCovered(int[][] ranges, int left, int right) {
        int[] num = new int[right+1];
        for(int[] a:ranges){
            for(int i=a[0];i<=a[1];i++){
               if(i<=right){
                   num[i]=1;
               }else {
                   break;
               }
            }
        }
        for(int i=left;i<=right;i++){
            if(num[i]!=1){
                return false;
            }
        }
        return true;
    }

    /**
     * 671. 二叉树中第二小的节点
     * 给定一个非空特殊的二叉树，每个节点都是正数，并且每个节点的子节点数量只能为 2 或 0。如果一个节点有两个子节点的话，那么该节点的值等于两个子节点中较小的一个。
     *
     * 更正式地说，root.val = min(root.left.val, root.right.val) 总成立。
     *
     * 给出这样的一个二叉树，你需要输出所有节点中的第二小的值。如果第二小的值不存在的话，输出 -1 。
     */
    int f ;
    int ans ;
    public int findSecondMinimumValue(TreeNode root) {
        if (root==null){
            return -1;
        }
        f = root.val;
        ans = -1;
        process(root);
        return ans;
    }
    private void process(TreeNode root){
        if(root==null){
            return;
        }
        if(ans!=-1&&root.val>=ans){
            return;
        }
        if(root.val>f){
            ans=root.val;
        }
        process(root.left);
        process(root.right);
    }

    /**
     * 863. 二叉树中所有距离为 K 的结点
     * 给定一个二叉树（具有根结点 root）， 一个目标结点 target ，和一个整数值 K 。
     *
     * 返回到目标结点 target 距离为 K 的所有结点的值的列表。 答案可以以任何顺序返回。
     */
    public List<Integer> distanceK(TreeNode root, TreeNode target, int k) {
        Map<TreeNode, TreeNode> map = new HashMap<>();
        Set<TreeNode> set =new HashSet<>();
        //dps，用哈希表保存每个节点的父节点信息
        dps863(root,map);
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(target);
        set.add(target);
        List<Integer> res = new ArrayList<>();
        //广度优先遍历，从target出发，向左右父节点寻找，用set记录已经路过的节点。
        while(k>0){
            k--;
            int size = queue.size();
            while(size>0){
                TreeNode now = queue.poll();
                TreeNode last = map.get(now);
                if(now.left!=null&&!set.contains(now.left)){
                    queue.add(now.left);
                    set.add(now.left);
                }
                if(now.right!=null&&!set.contains(now.right)){
                    queue.add(now.right);
                    set.add(now.right);
                }
                if(last!=null&&!set.contains(last)){
                    queue.add(last);
                    set.add(last);
                }
                size--;
            }
        }
        while(!queue.isEmpty()){
            res.add(queue.poll().val);
        }
        return res;
    }

    private void dps863(TreeNode root, Map<TreeNode, TreeNode> map){
        if(root==null){
            return;
        }
        if(root.left!=null){
            map.put(root.left,root);
        }
        if(root.right!=null){
            map.put(root.right,root);
        }
        dps863(root.left,map);
        dps863(root.right,map);
    }

    /**
     * 1104. 二叉树寻路
     * 在一棵无限的二叉树上，每个节点都有两个子节点，树中的节点 逐行 依次按 “之” 字形进行标记。
     *
     * 如下图所示，在奇数行（即，第一行、第三行、第五行……）中，按从左到右的顺序进行标记；
     *
     * 而偶数行（即，第二行、第四行、第六行……）中，按从右到左的顺序进行标记。
     */
    public List<Integer> pathInZigZagTree(int label) {
        int row = 1, rowStart = 1;
        while (rowStart * 2 <= label) {
            row++;
            rowStart *= 2;
        }
        if (row % 2 == 0) {
            label = getReverse(label, row);
        }
        List<Integer> path = new ArrayList<Integer>();
        while (row > 0) {
            if (row % 2 == 0) {
                path.add(getReverse(label, row));
            } else {
                path.add(label);
            }
            row--;
            label >>= 1;
        }
        Collections.reverse(path);
        return path;
    }

    public int getReverse(int label, int row) {
        return (1 << row - 1) + (1 << row) - 1 - label;
    }

    /**
     * 171. Excel表列序号
     * 给你一个字符串 columnTitle ，表示 Excel 表格中的列名称。返回该列名称对应的列序号。
     */
    public int titleToNumber(String columnTitle) {
        int res = 0;
        int size = columnTitle.length();
        for(int i=0;i<size;i++){
            char n = columnTitle.charAt(i);
            int a = n-'A'+1;
            res = res*26+a;
        }
        return res;
    }

    /**
     * 743. 网络延迟时间
     * 有 n 个网络节点，标记为 1 到 n。
     *
     * 给你一个列表 times，表示信号经过 有向 边的传递时间。 times[i] = (ui, vi, wi)，其中 ui 是源节点，vi 是目标节点， wi 是一个信号从源节点传递到目标节点的时间。
     *
     * 现在，从某个节点 K 发出一个信号。需要多久才能使所有节点都收到信号？如果不能使所有节点收到信号，返回 -1 。
     */
    int N = 110, M = 6010;
    int[][] w = new int[N][N];
    int INF = 0x3f3f3f3f;
    int n, k;
    public int networkDelayTime(int[][] ts, int _n, int _k) {
        n = _n; k = _k;
        // 初始化邻接矩阵
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                w[i][j] = w[j][i] = i == j ? 0 : INF;
            }
        }
        // 邻接矩阵存图
        for (int[] t : ts) {
            int u = t[0], v = t[1], c = t[2];
            w[u][v] = c;
        }
        // Floyd
        floyd();
        // 遍历答案
        int ans = 0;
        for (int i = 1; i <= n; i++) {
            ans = Math.max(ans, w[k][i]);
        }
        return ans >= INF / 2 ? -1 : ans;
    }
    void floyd() {
        for (int p = 1; p <= n; p++) {
            for (int i = 1; i <= n; i++) {
                for (int j = 1; j <= n; j++) {
                    w[i][j] = Math.min(w[i][j], w[i][p] + w[p][j]);
                }
            }
        }
    }

    /**
     * 581. 最短无序连续子数组
     * 给你一个整数数组 nums ，你需要找出一个 连续子数组 ，如果对这个子数组进行升序排序，那么整个数组都会变为升序排序。
     * 请你找出符合题意的 最短 子数组，并输出它的长度。
     */
    int MIN = -100005, MAX = 100005;
    public int findUnsortedSubarray(int[] nums) {
        int n = nums.length;
        int i = 0, j = n - 1;
        //找到单调递增的左右边界
        while (i < j && nums[i] <= nums[i + 1]) i++;
        while (i < j && nums[j] >= nums[j - 1]) j--;
        int l = i, r = j;
        int min = nums[i], max = nums[j];
        //从左边界遍历到右边界，不断调整左右边界的值
        for (int u = l; u <= r; u++) {
            if (nums[u] < min) {
                while (i >= 0 && nums[i] > nums[u]) i--;
                min = i >= 0 ? nums[i] : MIN;
            }
            if (nums[u] > max) {
                while (j < n && nums[j] < nums[u]) j++;
                max = j < n ? nums[j] : MAX;
            }
        }
        return j == i ? 0 : (j - 1) - (i + 1) + 1;
    }

    /**
     * 611. 有效三角形的个数
     * 给定一个包含非负整数的数组，你的任务是统计其中可以组成三角形三条边的三元组个数。
     */
    public int triangleNumber(int[] nums) {
        int len = nums.length;
        Arrays.sort(nums);
        int res = 0;
        for(int i=0;i<len-2;i++){
            for(int j=i+1;j<len-1;j++){
                int sum = nums[i]+nums[j];
                int r = len-1;
                int l = j;
                while(l<r){
                    int mid = (l+r+1)/2;
                    if(nums[mid]>=sum){
                        r = mid-1;
                    }else {
                        l = mid;
                    }
                }
                res+=r-j;
            }
        }
        return res;
    }

    /**
     * 413. 等差数列划分
     * 如果一个数列 至少有三个元素 ，并且任意两个相邻元素之差相同，则称该数列为等差数列。
     *
     * 例如，[1,3,5,7,9]、[7,7,7,7] 和 [3,-1,-5,-9] 都是等差数列。
     * 给你一个整数数组 nums ，返回数组 nums 中所有为等差数组的 子数组 个数。
     *
     * 子数组 是数组中的一个连续序列。
     */
    public int numberOfArithmeticSlices(int[] nums) {
        int len = nums.length;
        if(len<3){
            return 0;
        }
        int[] a = new int[len];
        for(int i=1;i<len;i++){
            a[i] = nums[i]-nums[i-1];
        }
        int res = 0;
        int tmp = 1;
        for(int i=2;i<len;i++){
            if(a[i]==a[i-1]){
                tmp++;
            }else {
                if(tmp!=1){
                    res+=tmp*(tmp-1)/2;
                }
                tmp=1;
            }
        }
        if(tmp!=1){
            res+=tmp*(tmp-1)/2;
        }
        return res;
    }

    /**
     * 443. 压缩字符串
     * 给你一个字符数组 chars ，请使用下述算法压缩：
     *
     * 从一个空字符串 s 开始。对于 chars 中的每组 连续重复字符 ：
     *
     * 如果这一组长度为 1 ，则将字符追加到 s 中。
     * 否则，需要向 s 追加字符，后跟这一组的长度。
     * 压缩后得到的字符串 s 不应该直接返回 ，需要转储到字符数组 chars 中。需要注意的是，如果组长度为 10 或 10 以上，则在 chars 数组中会被拆分为多个字符。
     *
     * 请在 修改完输入数组后 ，返回该数组的新长度。
     *
     * 你必须设计并实现一个只使用常量额外空间的算法来解决此问题。
     */
    public int compress(char[] chars) {
        StringBuilder sb = new StringBuilder();
        int i = 1;
        int len = chars.length;
        int sum = 1;
        if(i==len){
            return 1;
        }
        while(i<len){
            if(chars[i]==chars[i-1]){
                sum++;
            }else {
                sb.append(chars[i-1]);
                if(sum>1){
                    sb.append(sum);
                }
                sum = 1;
            }
            i++;
        }
        sb.append(chars[i-1]);
        if(sum>1){
            sb.append(sum);
        }
        for(int j=0;j<sb.length();j++){
            chars[j]=sb.charAt(j);
        }
        return sb.length();
    }

    /**
     * 789. 逃脱阻碍者
     * 你在进行一个简化版的吃豆人游戏。你从 [0, 0] 点开始出发，你的目的地是 target = [xtarget, ytarget] 。地图上有一些阻碍者，以数组 ghosts 给出，第 i 个阻碍者从 ghosts[i] = [xi, yi] 出发。所有输入均为 整数坐标 。
     *
     * 每一回合，你和阻碍者们可以同时向东，西，南，北四个方向移动，每次可以移动到距离原位置 1 个单位 的新位置。当然，也可以选择 不动 。所有动作 同时 发生。
     *
     * 如果你可以在任何阻碍者抓住你 之前 到达目的地（阻碍者可以采取任意行动方式），则被视为逃脱成功。如果你和阻碍者同时到达了一个位置（包括目的地）都不算是逃脱成功。
     *
     * 只有在你有可能成功逃脱时，输出 true ；否则，输出 false 。
     */
    public boolean escapeGhosts(int[][] ghosts, int[] target) {
        int min = Math.abs(target[0])+Math.abs(target[1]);
        int t = Integer.MAX_VALUE;
        for(int i=0;i<ghosts.length;i++){
            int[] g = ghosts[i];
            t = Math.min(t,Math.abs(g[0]-target[0])+Math.abs(g[1]-target[1]));
        }
        return min<t;
    }

    /**
     * 446. 等差数列划分 II - 子序列
     * 给你一个整数数组 nums ，返回 nums 中所有 等差子序列 的数目。
     *
     * 如果一个序列中 至少有三个元素 ，并且任意两个相邻元素之差相同，则称该序列为等差序列。
     *
     * 例如，[1, 3, 5, 7, 9]、[7, 7, 7, 7] 和 [3, -1, -5, -9] 都是等差序列。
     * 再例如，[1, 1, 2, 5, 7] 不是等差序列。
     * 数组中的子序列是从数组中删除一些元素（也可能不删除）得到的一个序列。
     *
     * 例如，[2,5,10] 是 [1,2,1,2,4,1,5,10] 的一个子序列。
     * 题目数据保证答案是一个 32-bit 整数。
     *
     *
     */
    public int numberOfArithmeticSlices2(int[] nums) {
        int ans = 0;
        int n = nums.length;
        Map<Long, Integer>[] f = new Map[n];
        for (int i = 0; i < n; ++i) {
            f[i] = new HashMap<Long, Integer>();
        }
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                long d = 1L * nums[i] - nums[j];
                int cnt = f[j].getOrDefault(d, 0);
                ans += cnt;
                f[i].put(d, f[i].getOrDefault(d, 0) + cnt + 1);
            }
        }
        return ans;
    }

    /**
     * 516. 最长回文子序列
     * 给你一个字符串 s ，找出其中最长的回文子序列，并返回该序列的长度。
     * 子序列定义为：不改变剩余字符顺序的情况下，删除某些字符或者不删除任何字符形成的一个序列。
     * 方法：区间dp
     * 通常区间 DP 问题都是，常见的基本流程为：
     * 从小到大枚举区间大小 len
     * 枚举区间左端点 l，同时根据区间大小 len 和左端点计算出区间右端点 r = l + len - 1
     * 通过状态转移方程求 f[l][r]的值
     */
    public int longestPalindromeSubseq(String s) {
        int l = s.length();
        //定义递归数组：dp[i][j]  为 在i-j范围内最长子序列的长度
        int[][] dp = new int[l][l];
        //在i-j范围的的最长子序列取决于
        //1:当i+1 - j-1范围内子序列长度等于回文串长度时 如果 s[i] = s[j] 则 dp[i][j]=dp[i+1][j-1] + 2
        //2:s[i] != s[j] dp[i][j] 取的是 dp[i+1][j] dp[i][j-1] 中的最大值，因为i与j不相等时，同时添加到回文串两边不会影响回文串的长度，所以只能加左边或者右边的数，然后取其中的最大值
        for(int i = l-1;i>=0;i--){
            dp[i][i] = 1;
            for(int j = i+1;j<l;j++){
                if(s.charAt(i)==s.charAt(j)){
                    dp[i][j] = dp[i+1][j-1]+2;
                }else {
                    dp[i][j] = Math.max(dp[i+1][j],dp[i][j-1]);
                }
            }
        }
        return dp[0][l-1];
    }

    /**
     * 233. 数字 1 的个数
     * 给定一个整数 n，计算所有小于等于 n 的非负整数中数字 1 出现的个数。
     */
    public int countDigitOne(int n) {
        int digit = 1, res = 0;
        int high = n / 10, cur = n % 10, low = 0;
        while(high != 0 || cur != 0) {
            if(cur == 0) res += high * digit;
            else if(cur == 1) res += high * digit + low + 1;
            else res += (high + 1) * digit;
            low += cur * digit;
            cur = high % 10;
            high /= 10;
            digit *= 10;
        }
        return res;
    }

    /**
     * 526. 优美的排列
     * 假设有从 1 到 N 的 N 个整数，如果从这 N 个数字中成功构造出一个数组，使得数组的第 i 位 (1 <= i <= N) 满足如下两个条件中的一个，我们就称这个数组为一个优美的排列。条件：
     * 第 i 位的数字能被 i 整除
     * i 能被第 i 位上的数字整除
     * 现在给定一个整数 N，请问可以构造多少个优美的排列？
     */
    int res = 0;
    public int countArrangement(int n) {
        int[] a = new int[n+1];
        for(int i=1;i<=n;i++){
            a[i] = i;
        }
        process(a,n);
        return res;
    }

    private void process(int[] a,int n){
        if(n==0){
            res++;
            return;
        }
        for(int i=1;i<a.length;i++){
            if(a[i]==-1){
                continue;
            }
            int t = a[i];
            if(isSuitable(a.length-n,a[i])){
                a[i]=-1;
                process(a,n-1);
            }else {
                continue;
            }
            a[i] = t;
        }
    }

    private Boolean isSuitable(int a,int b){
        if(a%b==0||b%a==0){
            return true;
        }else {
            return false;
        }
    }
    /**
     * 541. 反转字符串 II
     * 给定一个字符串 s 和一个整数 k，从字符串开头算起，每 2k 个字符反转前 k 个字符。
     * 如果剩余字符少于 k 个，则将剩余字符全部反转。
     * 如果剩余字符小于 2k 但大于或等于 k 个，则反转前 k 个字符，其余字符保持原样。
     */
    public String reverseStr(String s, int k) {
        int len = s.length();
        char[] c = s.toCharArray();
        int l = 0;
        while(l<len){
            if(l+k-1<len){
                swap(l,l+k-1,c);
            }else {
                swap(l,len-1,c);
            }
            l+=2*k;
        }
        return String.copyValueOf(c);
    }

    private void swap(int a,int b,char[] s){
        while(a<b){
            char t = s[a];
            s[a++] = s[b];
            s[b--] = t;
        }
    }


    /**
     * 1646. 获取生成数组中的最大值
     * 给你一个整数 n 。按下述规则生成一个长度为 n + 1 的数组 nums ：
     *
     * nums[0] = 0
     * nums[1] = 1
     * 当 2 <= 2 * i <= n 时，nums[2 * i] = nums[i]
     * 当 2 <= 2 * i + 1 <= n 时，nums[2 * i + 1] = nums[i] + nums[i + 1]
     * 返回生成数组 nums 中的 最大 值。
     */
    public int getMaximumGenerated(int n) {
        if(n<=1){
            return n;
        }
        int[] dp = new int[n+1];
        dp[0] = 0;
        dp[1] = 1;
        int max = 0;
        for(int i=2;i<=n;i++){
            if(i%2==0){
                dp[i] = dp[i/2];
            }else {
                dp[i] = dp[(i-1)/2] + dp[(i-1)/2+1];
            }
            max = Math.max(max,dp[i]);
        }
        return max;
    }

    /**
     * 787. K 站中转内最便宜的航班
     * 有 n 个城市通过一些航班连接。给你一个数组 flights ，其中 flights[i] = [fromi, toi, pricei] ，表示该航班都从城市 fromi 开始，以价格 toi 抵达 pricei。
     *
     * 现在给定所有的城市和航班，以及出发城市 src 和目的地 dst，你的任务是找到出一条最多经过 k 站中转的路线，使得从 src 到 dst 的 价格最便宜 ，并返回该价格。 如果不存在这样的路线，则输出 -1。
     */
    public int findCheapestPrice(int n, int[][] flights, int src, int dst, int k) {
        //建立hash表记录全部的点到点的记录
        HashMap<Integer,List<int[]>> map = new HashMap<>();
        for(int i=0;i<flights.length;i++){
            int[] f = flights[i];
            List t;
            if(map.containsKey(f[0])){
                 t = map.get(f[0]);
            }else {
                 t = new ArrayList();
            }
            t.add(new int[]{f[1],f[2]});
            map.put(f[0],t);
        }
        //开始广度度优先遍历
        //先把起始节点加入队列中
        Queue<int[]> queue = new LinkedList();
        queue.add(new int[]{src,0});
        //记录初始点到各个节点的最小花销
        HashMap<Integer,Integer> cost = new HashMap<>();
        while(!queue.isEmpty()&&k>=0){
            k--;
            int size = queue.size();
            for(int i=0;i<size;i++){
                int[] t = queue.poll();
                int sumPrice = t[1];
                if(map.containsKey(t[0])){
                    List<int[]> list = map.get(t[0]);
                    for(int[] s:list){

                        if(cost.containsKey(s[0])){
                            //判断节点有经过，如果有经过，判断花销是否有变小，是则更新cost，并把节点入列
                            if(cost.get(s[0])>sumPrice+s[1]){
                                queue.add(new int[]{s[0],sumPrice+s[1]});
                                cost.put(s[0],sumPrice+s[1]);
                            }
                        }else {
                            queue.add(new int[]{s[0],sumPrice+s[1]});
                            cost.put(s[0],sumPrice+s[1]);
                        }
                    }
                }
            }
        }
        return cost.get(dst)==null?-1:cost.get(dst);
    }


    /**
     * 797. 所有可能的路径
     * 给你一个有 n 个节点的 有向无环图（DAG），请你找出所有从节点 0 到节点 n-1 的路径并输出（不要求按特定顺序）
     *
     * 二维数组的第 i 个数组中的单元都表示有向图中 i 号节点所能到达的下一些节点，空就是没有下一个结点了。
     *
     * 译者注：有向图是有方向的，即规定了 a→b 你就不能从 b→a 。
     */
    public List<List<Integer>> allPathsSourceTarget(int[][] graph) {
        Queue<List<Integer>> queue =new LinkedList<>();
        List<Integer> list =new ArrayList<>();
        list.add(0);
        queue.add(list);
        List<List<Integer>> res = new ArrayList<>();
        while(!queue.isEmpty()){
            int size = queue.size();
            for(int i=0;i<size;i++){
                List<Integer> last = queue.poll();
                int[] a= graph[last.get(last.size()-1)];
                if(a!=null){
                    for(int j=0;j<a.length;j++){
                        List t = new ArrayList<>(last);
                        t.add(a[j]);
                        if(a[j]==graph.length-1){
                            res.add(t);
                        }else {
                            queue.add(t);
                        }
                    }
                }
            }
        }
        return res;
    }

    /**
     * 881. 救生艇
     * 第 i 个人的体重为 people[i]，每艘船可以承载的最大重量为 limit。
     *
     * 每艘船最多可同时载两人，但条件是这些人的重量之和最多为 limit。
     *
     * 返回载到每一个人所需的最小船数。(保证每个人都能被船载)。
     *
     *思路：贪心
     */
    public int numRescueBoats(int[] people, int limit) {
        Arrays.sort(people);
        int sum = 0;
        int l = 0;
        int r = people.length-1;
        while(l<=r){
            if(people[l]+people[r]<=limit){
                sum++;
                l++;
                r--;
            }else {
                sum++;
                r--;
            }
        }
        return sum;
    }

    /**
     * 1109. 航班预订统计
     * 这里有 n 个航班，它们分别从 1 到 n 进行编号。
     *
     * 有一份航班预订表 bookings ，表中第 i 条预订记录 bookings[i] = [firsti, lasti, seatsi] 意味着在从 firsti 到 lasti （包含 firsti 和 lasti ）的 每个航班 上预订了 seatsi 个座位。
     *
     * 请你返回一个长度为 n 的数组 answer，其中 answer[i] 是航班 i 上预订的座位总数。
     */
    public int[] corpFlightBookings(int[][] bookings, int n) {
        int[] res = new int[n];
        for(int i=0;i<bookings.length;i++){
            int[] tmp = bookings[i];
            for(int j=tmp[0];j<=tmp[1];j++){
                res[j-1]+=tmp[2];
            }
        }
        return res;
    }
}
