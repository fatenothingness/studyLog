package Topic;

import util.ListNode;
import util.TreeNode;

import java.util.*;

public class OneTopicEveryday {



    /**
     * 115. 不同的子序列
     */
    public int numDistinct(String s, String t) {
        char[] str = s.toCharArray();
        StringBuilder sb = new StringBuilder(t);
        int[][] dp = new int[s.length()][t.length()];
        return process(str,sb,0,dp);
    }

    public int process(char[] str,StringBuilder sb,int index,int[][] dp){
        int res = 0;
        if(sb.length()==0){
            return 1;
        }else {
            for(int i=index;i<str.length;i++){
                if(str[i]==sb.charAt(0)){
                    res+=process(str,sb.delete(0,1),i+1,dp);
                    sb.insert(0,str[i]);
                }
            }
            return res;
        }
    }


    /**
     * 92. 反转链表 II
     * 反转从位置 m 到 n 的链表。请使用一趟扫描完成反转
     */
    public ListNode reverseBetween(ListNode head, int left, int right) {
        //首先记录左节点的前一个节点作为开始节点（防止最左节点为头节点，先把节点值设为null）
        ListNode start = null;
        int i =1;
        ListNode l = head;
        while(i<left){
            start = l;
            l = l.next;
            i++;
        }
        //从左节点开始反转链表
        ListNode last = l;
        ListNode k = l.next;
        while(left<right){
            ListNode tmp = k.next;
            k.next = last;
            last = k;
            k = tmp;
            left++;
        }
        //将start的下一个节点指向反转链表的头节点，将反转链表的尾节点指向end节点
        if(start!=null){
            start.next = last;
            start = head;
        }else {
            start = last;
        }
        l.next = k;
        return start;
    }

    /**
     * 1603. 设计停车系统
     */
    class ParkingSystem {
        int big;
        int medium;
        int small;
        int[] arr;
        public ParkingSystem(int big, int medium, int small) {
            this.big = big;
            this.medium = medium;
            this.small = small;
            this.arr = new int[3];
        }

        public boolean addCar(int carType) {
            int value;
            switch (carType){
                case 1:
                    value = big;
                    break;
                case 2:
                    value = medium;
                    break;
                case 3:
                    value = small;
                    break;
                default:
                    value = 0;
            }
            if(arr[carType-1]<value){
                arr[carType-1]++;
                return true;
            }else {
                return false;
            }
        }
    }


    /**
     * 456. 132模式
     * 给你一个整数数组 nums ，数组中共有 n 个整数。132 模式的子序列 由三个整数 nums[i]、nums[j] 和 nums[k] 组成，并同时满足：i < j < k 和 nums[i] < nums[k] < nums[j]
     */
    public boolean find132pattern(int[] nums) {
        Stack<Integer> stack = new Stack<>();
        int k = Integer.MIN_VALUE;
        stack.push(nums[nums.length-1]);
        for(int i=nums.length-2;i>0;i--){
            if (nums[i] < k){
                return true;
            }
            while(!stack.isEmpty()&&nums[i]>stack.peek()){
                k = Math.max(k,stack.pop());
            }
                stack.push(nums[i]);
        }
        return false;
    }

    /**
     * 82. 删除排序链表中的重复元素 II
     * 存在一个按升序排列的链表，给你这个链表的头节点 head ，请你删除链表中所有存在数字重复情况的节点，只保留原始链表中 没有重复出现 的数字。
     */
    public ListNode deleteDuplicates(ListNode head) {
        int tmp = Integer.MIN_VALUE;
        ListNode res = new ListNode(Integer.MAX_VALUE,head);
        ListNode result = res;
        while(head!=null){
            if(head.next==null&&head.val>tmp){
                res.next = head;
                break;
            }
            if(head.next!=null&&head.next.val>head.val&&head.val!=tmp){
                res.next = head;
                res = res.next;
            }else {
                res.next = null;
            }
            tmp = head.val;
            head = head.next;
        }
        return result.next;
    }

    /**
     * 83. 删除排序链表中的重复元素
     * 存在一个按升序排列的链表，给你这个链表的头节点 head ，请你删除所有重复的元素，使每个元素 只出现一次 。
     * 返回同样按升序排列的结果链表。
     */

    public ListNode deleteDuplicates1(ListNode head) {
        ListNode tmp = new ListNode(Integer.MAX_VALUE,head);
        ListNode res = tmp;
        Integer last = Integer.MIN_VALUE;
        while(head!=null){
            if(head.val>last){
                tmp.next = head;
                tmp=tmp.next;
            }
            last = head.val;
            head=head.next;
            tmp.next=null;
        }
        return res.next;
    }

    /**
     *74. 搜索二维矩阵
     * 编写一个高效的算法来判断矩阵中，是否存在一个目标值。该矩阵具有如下特性：
     *
     * 每行中的整数从左到右按升序排列。
     * 每行的第一个整数大于前一行的最后一个整数。
     */
    //考察二分法基础写法，需要注意的点为：1 取中间值的时候需要最小+最大+1再除二 2 判断条件时，不包含等于的条件里，需要 = mid-1
    public boolean searchMatrix(int[][] matrix, int target) {
        int low = 0;
        int high = matrix.length-1;
        //先用二分判断行数
        while(low<high){
            int mid = (low+high+1)/2;
            if(matrix[mid][0]>target){
                high = mid-1;
            }else {
                low = mid;
            }
        }
        //取到行数后先判断第一个数是否等于目标数
        if(matrix[low][0]==target){
            return true;
        }
        //所在行进行二分查找
        int left = 0;
        int right = matrix[0].length-1;
        while(left<right){
            int mid = (left+right+1)/2;
            if(matrix[low][mid]>target){
                right = mid-1;
            }else if(matrix[low][mid]<target){
                left = mid;
            }else {
                return true;
            }
        }
        return false;
    }

    /**
     *90. 子集 II
     * 给你一个整数数组 nums ，其中可能包含重复元素，请你返回该数组所有可能的子集（幂集）。
     * 解集 不能 包含重复的子集。返回的解集中，子集可以按 任意顺序 排列。
     */
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        List<List<Integer>> ans = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        // 首先排序，让相同的两个元素排到一起去，便于去重
        Arrays.sort(nums);
        int n = nums.length;
        // 使用 visited 数组来记录哪一个元素在当前路径中被使用了
        boolean[] visited = new boolean[n];
        // 开始回溯
        backtrace(nums, 0, false, n,ans,path);
        return ans;
    }

    private void backtrace(int[] nums, int start, boolean visited, int n,List<List<Integer>> ans,List<Integer> path) {
        // 首先加入当前路径
        ans.add(new ArrayList<>(path));
        // 从 start 开始遍历每一个元素，尝试加入路径中
        for (int i = start; i < n; ++i) {
            // 如果当前元素和前一个元素相同，而且前一个元素没有被访问，说明前一个相同的元素在当前层已经被用过了
            if (i > 0 && nums[i - 1] == nums[i] && visited) continue;
            // 记录下来，用过了当前的元素
            path.add(nums[i]); // 放到路径中
            backtrace(nums, i + 1, true, n,ans,path); // 向下一个递归回溯
            path.remove(path.size() - 1);
            visited = false;
        }
    }

    /**
     *80. 删除有序数组中的重复项 II
     * 给你一个有序数组 nums ，请你 原地 删除重复出现的元素，使每个元素 最多出现两次 ，返回删除后数组的新长度。
     * 不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。
     */
    public int removeDuplicates(int[] nums) {
        int res = 0;
        boolean flag = true;
        for(int i=0;i<nums.length;i++){
            if(i==0){
                res++;
                continue;
            }
            if(nums[i]>nums[i-1]){
                nums[res] = nums[i];
                res++;
                flag = true;
            }
            else{
                if(flag){
                    nums[res] = nums[i];
                    res++;
                    flag = false;
                }else {
                    continue;
                }
            }
        }
        return res;
    }


    /**
     *81. 搜索旋转排序数组 II
     * 已知存在一个按非降序排列的整数数组 nums ，数组中的值不必互不相同。
     * 在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转 ，使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。例如， [0,1,2,4,4,4,5,6,6,7] 在下标 5 处经旋转后可能变为 [4,5,6,6,7,0,1,2,4,4] 。
     * 给你 旋转后 的数组 nums 和一个整数 target ，请你编写一个函数来判断给定的目标值是否存在于数组中。如果 nums 中存在这个目标值 target ，则返回 true ，否则返回 false 。
     * @param nums
     * @return
     */
    public boolean search(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return false;
        }
        int start = 0;
        int end = nums.length - 1;
        int mid;
        while (start <= end) {
            mid = start + (end - start) / 2;
            if (nums[mid] == target) {
                return true;
            }
            if (nums[start] == nums[mid]) {
                start++;
                continue;
            }
            //前半部分有序
            if (nums[start] < nums[mid]) {
                //target在前半部分
                if (nums[mid] > target && nums[start] <= target) {
                    end = mid - 1;
                } else {  //否则，去后半部分找
                    start = mid + 1;
                }
            } else {
                //后半部分有序
                //target在后半部分
                if (nums[mid] < target && nums[end] >= target) {
                    start = mid + 1;
                } else {  //否则，去后半部分找
                    end = mid - 1;

                }
            }
        }
        return false;
    }

    /**
     * 179. 最大数
     * 给定一组非负整数 nums，重新排列每个数的顺序（每个数不可拆分）使之组成一个最大的整数。
     *
     * 注意：输出结果可能非常大，所以你需要返回一个字符串而不是整数。
     */
    public String largestNumber(int[] nums) {
        for(int i=0;i<nums.length-1;i++){
            for(int j=0;j<nums.length-1-i;j++){
                if(!contrast(nums[j],nums[j+1])){
                    int tmp = nums[j];
                    nums[j] = nums[j+1];
                    nums[j+1] = tmp;
                }
            }
        }
        if(nums[0]==0){
            return String.valueOf(0);
        }
        StringBuilder sb = new StringBuilder();
        for(int a=0;a<nums.length;a++){
            sb.append(nums[a]);
        }
        return sb.toString();
    }
    public boolean contrast(int a ,int b) {
        char[] sa = (String.valueOf(a)+ b).toCharArray();
        char[] sb = (String.valueOf(b)+ a).toCharArray();
        int i = 0;
        while (i < sa.length) {
            if (sa[i] == sb[i]) {
                i++;
            } else if (sa[i] > sb[i]) {
                return true;
            } else {
                return false;
            }
        }
        return true;
    }

    /**
     * 783. 二叉搜索树节点最小距离
     * 给你一个二叉搜索树的根节点 root ，返回 树中任意两不同节点值之间的最小差值 。
     */
    public int minDiffInBST(TreeNode root) {
        List<Integer> list = new ArrayList<>();
        dp(root,list);
        int res = Integer.MAX_VALUE;
        for(int i=0;i<list.size()-1;i++){
            res = Math.min(res,list.get(i+1)-list.get(i));
        }
        return res;
    }
    public void dp(TreeNode root, List<Integer> list){
        if(root==null){
            return;
        }
        dp(root.left,list);
        list.add(root.val);
        dp(root.right,list);
    }

    /**
     * 213. 打家劫舍 II
     * 你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都 围成一圈 ，这意味着第一个房屋和最后一个房屋是紧挨着的。同时，相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警 。
     * 给定一个代表每个房屋存放金额的非负整数数组，计算你 在不触动警报装置的情况下 ，能够偷窃到的最高金额。
     */
    public int rob(int[] nums) {
        if(nums.length == 0) return 0;
        if(nums.length == 1) return nums[0];
        return Math.max(myRob(Arrays.copyOfRange(nums, 0, nums.length - 1)),
                myRob(Arrays.copyOfRange(nums, 1, nums.length)));
    }
    public int myRob(int[] nums) {
        //两个指针分别指向上一个和上上一个值；
        int res=0;
        int pre=0;
        int tmp;
        for(int num : nums) {
            tmp = res;
            res = Math.max(pre + num, res);
            pre = tmp;
        }
        return res;
    }
    /**
     * 39. 组合总和
     * 给定一个无重复元素的数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
     * candidates 中的数字可以无限制重复被选取
     */
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        dp(candidates,0,target,res,new ArrayList<>());
        return res;
    }
    public void dp(int[] candidates, int begin,int target,List<List<Integer>> res,List<Integer> tmp){
        if(target==0){
            res.add(CollectionCopy(tmp));
        }else if(target>0){
            for(int i=begin;i<candidates.length;i++){
                tmp.add(candidates[i]);
                dp(candidates,i,target-candidates[i],res,tmp);
                tmp.remove(tmp.size()-1);
            }
        }else {
            return;
        }
    }

    public List<Integer> CollectionCopy(List<Integer> tmp){
        List<Integer> res = new ArrayList<>();
        for(Integer t:tmp){
            res.add(t);
        }
        return res;
    }

    /**
     * 87. 扰乱字符串
     * 使用下面描述的算法可以扰乱字符串 s 得到字符串 t ：
     * 如果字符串的长度为 1 ，算法停止
     * 如果字符串的长度 > 1 ，执行下述步骤：
     * 在一个随机下标处将字符串分割成两个非空的子字符串。即，如果已知字符串 s ，则可以将其分成两个子字符串 x 和 y ，且满足 s = x + y 。
     * 随机 决定是要「交换两个子字符串」还是要「保持这两个子字符串的顺序不变」。即，在执行这一步骤之后，s 可能是 s = x + y 或者 s = y + x 。
     * 在 x 和 y 这两个子字符串上继续从步骤 1 开始递归执行此算法。
     * 给你两个 长度相等 的字符串 s1 和 s2，判断 s2 是否是 s1 的扰乱字符串。如果是，返回 true ；否则，返回 false 。
     */
    public boolean isScramble(String s1, String s2) {
        if (s1.equals(s2)) return true;
        if (s1.length() != s2.length()) return false;
        int n = s1.length();
        char[] cs1 = s1.toCharArray(), cs2 = s2.toCharArray();
        boolean[][][] f = new boolean[n][n][n + 1];

        // 先处理长度为 1 的情况
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                f[i][j][1] = cs1[i] == cs2[j];
            }
        }

        // 再处理其余长度情况
        for (int len = 2; len <= n; len++) {
            for (int i = 0; i <= n - len; i++) {
                for (int j = 0; j <= n - len; j++) {
                    for (int k = 1; k < len; k++) {
                        boolean a = f[i][j][k] && f[i + k][j + k][len - k];
                        boolean b = f[i][j + len - k][k] && f[i + k][j][len - k];
                        if (a || b) {
                            f[i][j][len] = true;
                        }
                    }
                }
            }
        }
        return f[0][0][n];
    }

    /**
     * 42. 接雨水
     * 给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。
     * @param height
     * @return
     */
    public int trap(int[] height) {
        int res = 0;
        Stack<Integer> stack = new Stack<>();
        for(int i =0;i<height.length;i++){
            while(!stack.isEmpty()&&height[stack.peek()]<height[i]){
                int cur = stack.peek();
                stack.pop();
                if(stack.isEmpty()){
                    break;
                }
                int left = stack.peek();
                int right = i;
                int h = Math.min(height[left],height[right])-height[cur];
                res += (right-left-1)*h;
            }
            stack.push(i);
        }
        return res;
    }

    /**
     * 27. 移除元素
     * 给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度。
     *
     * 不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并 原地 修改输入数组。
     *
     * 元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。
     */
    public int removeElement(int[] nums, int val) {
        int len = nums.length;
        int sum = 0;
        int i=0;
        int tmp;
        while(i<len-sum){
            if(nums[i]==val){
                tmp = nums[len-sum-1];
                nums[len-sum-1] = val;
                nums[i] = tmp;
                sum++;
            }else {
                i++;
            }
        }
        return len-sum;
    }
    /**
     * 91. 解码方法
     * 一条包含字母 A-Z 的消息通过以下映射进行了 编码 ：
     *
     * 'A' -> 1
     * 'B' -> 2
     * ...
     * 'Z' -> 26
     * 要 解码 已编码的消息，所有数字必须基于上述映射的方法，反向映射回字母（可能有多种方法）。例如，"11106" 可以映射为：
     *
     * "AAJF" ，将消息分组为 (1 1 10 6)
     * "KJF" ，将消息分组为 (11 10 6)
     * 注意，消息不能分组为  (1 11 06) ，因为 "06" 不能映射为 "F" ，这是由于 "6" 和 "06" 在映射中并不等价。
     *
     * 给你一个只含数字的 非空 字符串 s ，请计算并返回 解码 方法的 总数 。
     *
     * 题目数据保证答案肯定是一个 32 位 的整数。
     *
     *
     */
    public int numDecodings(String s) {
        int n = s.length();
        s = " " + s;
        char[] cs = s.toCharArray();
        int[] f = new int[n + 1];
        f[0] = 1;
        for (int i = 1; i <= n; i++) {
            // a : 代表「当前位置」单独形成 item
            // b : 代表「当前位置」与「前一位置」共同形成 item
            int a = cs[i] - '0', b = (cs[i - 1] - '0') * 10 + (cs[i] - '0');
            // 如果 a 属于有效值，那么 f[i] 可以由 f[i - 1] 转移过来
            if (1 <= a && a <= 9) f[i] = f[i - 1];
            // 如果 b 属于有效值，那么 f[i] 可以由 f[i - 2] 或者 f[i - 1] & f[i - 2] 转移过来
            if (10 <= b && b <= 26) f[i] += f[i - 2];
        }
        return f[n];
    }

    /**
     * 363. 矩形区域不超过 K 的最大数值和
     * 给你一个 m x n 的矩阵 matrix 和一个整数 k ，找出并返回矩阵内部矩形区域的不超过 k 的最大数值和。
     * 题目数据保证总会存在一个数值和不超过 k 的矩形区域
     */

    public int maxSumSubmatrix(int[][] matrix, int k) {
        //row 为行数（上下边界），cols为列数（左右边界）
        int rows = matrix.length, cols = matrix[0].length, max = Integer.MIN_VALUE;
        // O(cols ^ 2 * rows)
        for (int l = 0; l < cols; l++) { // 枚举左边界
            //初始化滚动数组
            int[] rowSum = new int[rows]; // 左边界改变才算区域的重新开始
            for (int r = l; r < cols; r++) { // 枚举右边界
                //从上到下开始累加左右边界的的和
                for (int i = 0; i < rows; i++) { // 按每一行累计到 dp
                    rowSum[i] += matrix[i][r];
                }
                // 求 rowSum 连续子数组 的 和
                // 和 尽量大，但不大于 k
                max = Math.max(max, dpmax(rowSum, k));
            }
        }
        return max;
    }
    private int dpmax(int[] arr, int k) {
        // O(rows ^ 2)
        int max = Integer.MIN_VALUE;
        for (int l = 0; l < arr.length; l++) {
            int sum = 0;
            for (int r = l; r < arr.length; r++) {
                sum += arr[r];
                if (sum > max && sum <= k) max = sum;
            }
        }
        return max;
    }

/**
 *1011. 在 D 天内送达包裹的能力
 * 传送带上的包裹必须在 D 天内从一个港口运送到另一个港口。
 *
 * 传送带上的第 i 个包裹的重量为 weights[i]。每一天，我们都会按给出重量的顺序往传送带上装载包裹。我们装载的重量不会超过船的最大运载重量。
 *
 * 返回能在 D 天内将传送带上的所有包裹送达的船的最低运载能力。
 */
    public int shipWithinDays(int[] weights, int D) {
        int sum = 0;
        int min = 0;
        for(int i=0;i<weights.length;i++){
            sum+=weights[i];
            if(min<weights[i]){
                min = weights[i];
            }
        }
        //在min 到 sum 之间寻找刚好能满足D天的最小值
        while(min<sum){
            int mid = (min+sum)>>1;
            if(isSatisfy(weights,mid,D)){
                sum = mid;
            }else {
                min = mid+1;
            }
        }
        return min;
    }

    public boolean isSatisfy(int[] weights,int s,int D) {
        int sum = 0;
        int res = 1;
        for(int i=0;i<weights.length;i++){
            sum+=weights[i];
            if(sum>s){
                res++;
                sum = weights[i];
            }
            if(res>D){
                return false;
            }
        }
        return true;
    }

    /**
     * 872. 叶子相似的树
     * 请考虑一棵二叉树上所有的叶子，这些叶子的值按从左到右的顺序排列形成一个 叶值序列 。
     */
    public boolean leafSimilar(TreeNode root1, TreeNode root2) {
        List<Integer> r1 = new ArrayList<>();
        List<Integer> r2 = new ArrayList<>();
        leafDp(root1,r1);
        leafDp(root2,r2);
        if(r1.size()!=r2.size()){
            return false;
        }else {
            for(int i=0;i<r1.size();i++){
                if(r1.get(i)!=r2.get(i)){
                    return  false;
                }
            }
            return  true;
        }
    }

    public void leafDp(TreeNode root, List<Integer> list){
        if(root.left==null&&root.right==null){
            list.add(root.val);
            return;
        }
        if(root.right!=null){
            leafDp(root.left,list);
        }
        if(root.right!=null){
            leafDp(root.right,list);
        }
    }

    public void rotate(int[][] matrix) {
        if(matrix.length==0||matrix.length!=matrix[0].length){
            return;
        }
        int nums = matrix.length;
        int times = 0;
        while(times<=(nums>>1)){
            //每一层划分成4份，当前层的长度比上一层少2
            int len = nums-(times<<1);
            for(int i=0;i<len-1;++i){
                int tmp = matrix[times][times+i];
                //左下角换到左上角
                matrix[times][times + i] = matrix[times + len - i - 1][times];
                //右下角换到左下角
                matrix[times + len - i - 1][times] = matrix[times + len - 1][times + len - i - 1];
                //右上角换到右下角
                matrix[times + len - 1][times + len - i - 1] = matrix[times + i][times + len - 1];
                //左上角换到右上角
                matrix[times + i][times + len - 1] = tmp;
            }
            ++times;
        }
    }

    /**
     * 1734. 解码异或后的排列
     * 给你一个整数数组 perm ，它是前 n 个正整数的排列，且 n 是个 奇数 。
     *
     * 它被加密成另一个长度为 n - 1 的整数数组 encoded ，满足 encoded[i] = perm[i] XOR perm[i + 1] 。比方说，如果 perm = [1,3,2] ，那么 encoded = [2,1] 。
     *
     * 给你 encoded 数组，请你返回原始数组 perm 。题目保证答案存在且唯一。
     */

    public int[] decode(int[] encoded) {
        int n = encoded.length + 1;
        int[] ans = new int[n];
        //eccoded[1] = ans[1]^ans[2]  encoded[3] = ans[3]^ans[4]
        // 将encoded隔一位做异或，求得除了 ans[n - 1] 的所有异或结果(即ans[1]^ans[2]^ans[3]....^ans[n-2])
        int a = 0;
        for (int i = 0; i < n - 1; i += 2) a ^= encoded[i];
        // 求得 ans 的所有异或结果(ans[1]^ans[2]^ans[3]....^ans[n-1])
        int b = 0;
        for (int i = 1; i <= n; i++) b ^= i;
        // 求得 ans[n - 1] 后，从后往前做    ( a^b 则得到 ans[n-1],相同数异或得0，异或满足交换律)
        ans[n - 1] = a ^ b;
        for (int i = n - 2; i >= 0; i--) {
            ans[i] = encoded[i] ^ ans[i + 1];
        }
        return ans;
    }


    /**
     * 1442. 形成两个异或相等数组的三元组数目
     * 给你一个整数数组 arr 。
     * 现需要从数组中取三个下标 i、j 和 k ，其中 (0 <= i < j <= k < arr.length) 。
     * a 和 b 定义如下：
     * a = arr[i] ^ arr[i + 1] ^ ... ^ arr[j - 1]
     * b = arr[j] ^ arr[j + 1] ^ ... ^ arr[k]
     * 注意：^ 表示 按位异或 操作。
     * 请返回能够令 a == b 成立的三元组 (i, j , k) 的数目。
     */
    public int countTriplets(int[] arr) {
        int[] dp =new int[arr.length];
        dp[0] = arr[0];
        for(int i=1;i<arr.length;i++){
            dp[i]=dp[i-1]^arr[i];
        }
        int a,b;
        int res = 0;
        for(int i=0;i<arr.length-1;i++){
            for(int j=i+1;j<arr.length;j++){
                for(int k=j;k<arr.length;k++){
                    a=getSum(dp,i-1,j-1);
                    b=getSum(dp,j-1,k);
                    if(a==b){
                        res++;
                    }
                }
            }
        }
        return res;

    }
    private int getSum(int[] dp,int a,int b){
        if(a<0){
            return dp[b];
        }else {
            return dp[a]^dp[b];
        }
    }


    /**
     * 1738. 找出第 K 大的异或坐标值
     * 给你一个二维矩阵 matrix 和一个整数 k ，矩阵大小为 m x n 由非负整数组成。
     *
     * 矩阵中坐标 (a, b) 的 值 可由对所有满足 0 <= i <= a < m 且 0 <= j <= b < n 的元素 matrix[i][j]（下标从 0 开始计数）执行异或运算得到。
     *
     * 请你找出 matrix 的所有坐标中第 k 大的值（k 的值从 1 开始计数）。
     */
    public int kthLargestValue(int[][] matrix, int k) {
        int m = matrix.length;
        int n = matrix[0].length;
        int[][] dp = new int[m + 1][n + 1];
        PriorityQueue<Integer> q = new PriorityQueue<>(k, new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return o1-o2;
            }
        });
        for(int i=1;i<=m;i++){
            for(int j=1;j<=n;j++){
                dp[i][j] = dp[i-1][j]^dp[i][j-1]^dp[i-1][j-1]^matrix[i-1][j-1];
                if(q.size()<k){
                    q.add(dp[i][j]);
                }else {
                    if(dp[i][j]>q.peek()){
                        q.poll();
                        q.add(dp[i][j]);
                    }
                }
            }
        }
        return q.peek();
    }
    /**
     * 692. 前K个高频单词
     * 给一非空的单词列表，返回前 k 个出现次数最多的单词。
     * 返回的答案应该按单词出现频率由高到低排序。如果不同的单词有相同出现频率，按字母顺序排序。
     */
    class Words{
        public Integer sum;
        public String word;
        public Words(){

        }
        public Words(Integer sum,String word){
            this.sum = sum;
            this.word = word;
        }
    }
    private boolean compare(String o1,String o2){
        int size = Math.min(o1.length(),o2.length());
        int i=0;
        while(i<size){
            if (o1.charAt(i)-'a'==o2.charAt(i)-'a'){
                i++;
            }else if(o1.charAt(i)-'a'>o2.charAt(i)-'a'){
                return true;
            }else {
                return false;
            }
        }
        return o1.length()>o2.length();
    }
    public List<String> topKFrequent(String[] words, int k) {
        PriorityQueue<Words> priorityQueue = new PriorityQueue<>(k, (o1, o2) -> {
            if(o1.sum ==o2.sum){
                if(compare(o1.word,o2.word)){
                    return -1;
                }else {
                    return 1;
                }
            }else {
                return o1.sum-o2.sum;
            }
        });
        Map<String,Words> map = new HashMap<>();
        for(String s:words){
            if(map.containsKey(s)){
                map.get(s).sum++;
            }else {
                map.put(s,new Words(1,s));
            }
        }
        for(String s:map.keySet()){
            Words now = priorityQueue.peek();
            if(priorityQueue.size()<k){
                priorityQueue.add(map.get(s));
            }else {
                if(now.sum<map.get(s).sum){
                    priorityQueue.poll();
                    priorityQueue.add(map.get(s));
                }else if(now.sum==map.get(s).sum){
                    if(compare(now.word,s)){
                        priorityQueue.poll();
                        priorityQueue.add(map.get(s));
                    }
                }
            }
        }
        List<String> res = new ArrayList<>();
        while(!priorityQueue.isEmpty()){
            res.add(priorityQueue.poll().word);
        }
        Collections.reverse(res);
        return res;
    }


    /**
     * 810. 黑板异或游戏
     * 黑板上写着一个非负整数数组 nums[i] 。Alice 和 Bob 轮流从黑板上擦掉一个数字，Alice 先手。如果擦除一个数字后，剩余的所有数字按位异或运算得出的结果等于 0 的话，当前玩家游戏失败。 (另外，如果只剩一个数字，按位异或运算得到它本身；如果无数字剩余，按位异或运算结果为 0。）
     *
     * 换种说法就是，轮到某个玩家时，如果当前黑板上所有数字按位异或运算结果等于 0，这个玩家获胜。
     *
     * 假设两个玩家每步都使用最优解，当且仅当 Alice 获胜时返回 true。
     * @param nums
     * @return
     */
    public boolean xorGame(int[] nums) {
        int sum = 0;
        for(int i:nums){
            sum^=i;
        }
        return sum==0 || nums.length%2==0;
    }

    /**
     * 461. 汉明距离
     * 两个整数之间的汉明距离指的是这两个数字对应二进制位不同的位置的数目。
     *
     * 给出两个整数 x 和 y，计算它们之间的汉明距离。
     */
    //一个数和自己的负数做与操作，能得到最低位1对应的值
    int lowbit(int x) {
        return x & -x;
    }
    public int hammingDistance(int x, int y) {
        int ans = 0;
        for (int i = x ^ y; i > 0; i -= lowbit(i)) ans++;
        return ans;
    }

    /**
     * 1744. 你能在你最喜欢的那天吃到你最喜欢的糖果吗？
     * 给你一个下标从 0 开始的正整数数组 candiesCount ，其中 candiesCount[i] 表示你拥有的第 i 类糖果的数目。同时给你一个二维数组 queries ，其中 queries[i] = [favoriteTypei, favoriteDayi, dailyCapi] 。
     *
     * 你按照如下规则进行一场游戏：
     *
     * 你从第 0 天开始吃糖果。
     * 你在吃完 所有 第 i - 1 类糖果之前，不能 吃任何一颗第 i 类糖果。
     * 在吃完所有糖果之前，你必须每天 至少 吃 一颗 糖果。
     * 请你构建一个布尔型数组 answer ，满足 answer.length == queries.length 。answer[i] 为 true 的条件是：在每天吃 不超过 dailyCapi 颗糖果的前提下，你可以在第 favoriteDayi 天吃到第 favoriteTypei 类糖果；否则 answer[i] 为 false 。注意，只要满足上面 3 条规则中的第二条规则，你就可以在同一天吃不同类型的糖果。
     *
     * 请你返回得到的数组 answer 。
     */
    public boolean[] canEat(int[] cs, int[][] qs) {
        int n = qs.length, m = cs.length;
        boolean[] ans = new boolean[n];
        long[] sum = new long[m + 1];
        for (int i = 1; i <= m; i++) sum[i] = sum[i - 1] + cs[i - 1];
        for (int i = 0; i < n; i++) {
            int t = qs[i][0], d = qs[i][1] + 1, c = qs[i][2];
            long a = sum[t] / c + 1, b = sum[t + 1];
            ans[i] = a <= d && d <= b;
        }
        return ans;
    }
    /**
     * 523. 连续的子数组和
     * 给你一个整数数组 nums 和一个整数 k ，编写一个函数来判断该数组是否含有同时满足下述条件的连续子数组：
     *
     * 子数组大小 至少为 2 ，且
     * 子数组元素总和为 k 的倍数。
     * 如果存在，返回 true ；否则，返回 false 。
     *
     * 如果存在一个整数 n ，令整数 x 符合 x = n * k ，则称 x 是 k 的一个倍数。
     */
    public boolean checkSubarraySum(int[] nums, int k) {
        int[] sum = new int[nums.length];
        sum[0] = nums[0];
        for(int i=1;i<nums.length;i++){
            sum[i]=sum[i-1]+nums[i];
            if(sum[i]%k==0){
                return true;
            }
        }
        Set<Integer> set = new HashSet<>();
        for(int j=2;j<nums.length;j++){
            set.add(sum[j-2]%k);
            if(set.contains(sum[j]%k)){
                return true;
            }
        }
        return false;
    }

    /**
     * 525. 连续数组
     * 给定一个二进制数组 nums , 找到含有相同数量的 0 和 1 的最长连续子数组，并返回该子数组的长度。
     */

    public int findMaxLength(int[] nums) {
        int n = nums.length;
        int[] sum = new int[n + 1];
        for (int i = 1; i <= n; i++){
            sum[i] = sum[i - 1] + (nums[i - 1] == 1 ? 1 : -1);
        }
        int ans = 0;
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 2; i <= n; i++) {
            if (!map.containsKey(sum[i - 2])) map.put(sum[i - 2], i - 2);
            if (map.containsKey(sum[i])) ans = Math.max(ans, i - map.get(sum[i]));
        }
        return ans;
    }

    /**
     * 160. 相交链表
     * 给你两个单链表的头节点 headA 和 headB ，请你找出并返回两个单链表相交的起始节点。如果两个链表没有交点，返回 null 。
     */
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode a = headA;
        ListNode b = headB;
        while(a!=null&&b!=null){
            if(a==b){
                return a;
            }else {
                a=a.next;
                b=b.next;
            }
        }
        if(a!=null){
            while(a!=null){
                headA=headA.next;
                a=a.next;
            }
        }else if(b!=null){
            while(b!=null){
                headB=headB.next;
                b=b.next;
            }
        }else {
            return null;
        }
        while(headA!=null&&headB!=null){
            if(headA==headB){
                return headA;
            }else {
                headA=headA.next;
                headB=headB.next;
            }
        }
        return null;
    }

    /**
     * 203. 移除链表元素
     * 给你一个链表的头节点 head 和一个整数 val ，请你删除链表中所有满足 Node.val == val 的节点，并返回 新的头节点 。
     * @param head
     * @param val
     * @return
     */
    public ListNode removeElements(ListNode head, int val) {
        ListNode a = head;
        ListNode b = new ListNode(0,head);
        ListNode res = b;
        while(a!=null){
            if(a.val==val){
                a=a.next;
                b.next = a;
            }else {
                a=a.next;
                b=b.next;
            }
        }
        return  res.next;
    }

    /**
     * 494. 目标和
     * 给你一个整数数组 nums 和一个整数 target 。
     *
     * 向数组中的每个整数前添加 '+' 或 '-' ，然后串联起所有整数，可以构造一个 表达式 ：
     *
     * 例如，nums = [2, 1] ，可以在 2 之前添加 '+' ，在 1 之前添加 '-' ，然后串联起来得到表达式 "+2-1" 。
     * 返回可以通过上述方法构造的、运算结果等于 target 的不同 表达式 的数目。
     */
    public int findTargetSumWays(int[] nums, int t) {
        int[]dp = new int[nums.length];
        return dfs(nums, t, 0, 0);
    }
    int dfs(int[] nums, int t, int u, int cur) {
        //如果目标值等于和返回1
        if (u == nums.length) {
            return cur == t ? 1 : 0;
        }
        //当前数取正号
        int left = dfs(nums, t, u + 1, cur + nums[u]);
        //当前数取负号
        int right = dfs(nums, t, u + 1, cur - nums[u]);

        return left + right;
    }

    /**
     * 1049. 最后一块石头的重量 II
     * 有一堆石头，用整数数组 stones 表示。其中 stones[i] 表示第 i 块石头的重量。
     *
     * 每一回合，从中选出任意两块石头，然后将它们一起粉碎。假设石头的重量分别为 x 和 y，且 x <= y。那么粉碎的可能结果如下：
     *
     * 如果 x == y，那么两块石头都会被完全粉碎；
     * 如果 x != y，那么重量为 x 的石头将会完全粉碎，而重量为 y 的石头新重量为 y-x。
     * 最后，最多只会剩下一块 石头。返回此石头 最小的可能重量 。如果没有石头剩下，就返回 0。
     */
    public int lastStoneWeightII(int[] ss) {
        int n = ss.length;
        int sum = 0;
        for (int i : ss) sum += i;
        int t = sum / 2;
        int[] f = new int[t + 1];
        for (int i = 1; i <= n; i++) {
            int x = ss[i - 1];
            for (int j = t; j >= x; j--) {
                f[j] = Math.max(f[j], f[j - x] + x);
            }
        }
        return Math.abs(sum - f[t] - f[t]);
    }


/**
 * 879. 盈利计划
 * 集团里有 n 名员工，他们可以完成各种各样的工作创造利润。
 *
 * 第 i 种工作会产生 profit[i] 的利润，它要求 group[i] 名成员共同参与。如果成员参与了其中一项工作，就不能参与另一项工作。
 *
 * 工作的任何至少产生 minProfit 利润的子集称为 盈利计划 。并且工作的成员总数最多为 n 。
 *
 * 有多少种计划可以选择？因为答案很大，所以 返回结果模 10^9 + 7 的值。
 */
int mod = (int)1e9+7;
    public int profitableSchemes(int n, int min, int[] gs, int[] ps) {
        int m = gs.length;
        //定义三维dp数组，前i种工作中j个员工在min利润下的计划数。
        long[][][] f = new long[m + 1][n + 1][min + 1];
        //在最低利润为0且没有工作的情况下，不管多少员工的计划数都为1
        for (int i = 0; i <= n; i++) f[0][i][0] = 1;
        for (int i = 1; i <= m; i++) {
            //遍历工作 a为当前遍历的工作所需人数，b为当前工作的利润
            int a = gs[i - 1], b = ps[i - 1];
            for (int j = 0; j <= n; j++) {
                //遍历员工数
                for (int k = 0; k <= min; k++) {
                    //如果当前工作不够人数开始
                    f[i][j][k] = f[i - 1][j][k];
                    //如果当前员工数大于当前工作所需人数
                    if (j >= a) {
                        int u = Math.max(k - b, 0);
                        f[i][j][k] += f[i - 1][j - a][u];
                        if (f[i][j][k] >= mod) f[i][j][k] -= mod;
                    }
                }
            }
        }
        return (int)f[m][n][min];
    }

    /**
     * 518. 零钱兑换 II
     * 给定不同面额的硬币和一个总金额。写出函数来计算可以凑成总金额的硬币组合数。假设每一种面额的硬币有无限个。
     */

    public int change(int amount, int[] coins) {
        int[] dp = new int[amount + 1];
        dp[0] = 1; // 0元有一种, 那就是啥币也不拿
        for (int i = 0; i < coins.length; i++) { // 01背包
            for (int j = coins[i]; j <= amount; j++) {
                dp[j] += dp[j - coins[i]]; // 组合, 累积
            }
        }
        return dp[amount];
    }

    /**
     * 931. 下降路径最小和
     * 给你一个 n x n 的 方形 整数数组 matrix ，请你找出并返回通过 matrix 的下降路径 的 最小和 。
     * 下降路径 可以从第一行中的任何元素开始，并从每一行中选择一个元素。
     * 在下一行选择的元素和当前行所选元素最多相隔一列（即位于正下方或者沿对角线向左或者向右的第一个元素）。
     * 具体来说，位置 (row, col) 的下一个元素应当是 (row + 1, col - 1)、(row + 1, col) 或者 (row + 1, col + 1) 。
     */
    public int minFallingPathSum(int[][] matrix) {
        int n = matrix.length;
        int m = matrix[0].length;
        int[][] dp = new int[n][m];
        for(int i=0;i<m;i++) dp[0][i]=matrix[0][i];
        for(int i=1;i<n;i++){
            for(int j=0;j<m;j++){
                int v = matrix[i][j];
                int left=j-1>=0?dp[i-1][j-1]:Integer.MAX_VALUE;
                int up = dp[i-1][j];
                int right = j+1<n?dp[i-1][j+1]:Integer.MAX_VALUE;
                dp[i][j] = Math.min(left,Math.min(up,right))+v;
            }
        }
        int res = Integer.MAX_VALUE;
        for(int i =0;i<m;i++) res = Math.min(res,dp[n-1][i]);
        return res;
    }

    /**
     * 279. 完全平方数
     * 给定正整数 n，找到若干个完全平方数（比如 1, 4, 9, 16, ...）使得它们的和等于 n。你需要让组成和的完全平方数的个数最少。
     *
     * 给你一个整数 n ，返回和为 n 的完全平方数的 最少数量 。
     *
     * 完全平方数 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。例如，1、4、9 和 16 都是完全平方数，而 3 和 11 不是。
     *
     *
     */

    public int numSquares(int n) {
        int[] f = new int[n + 1];
        Arrays.fill(f, 0x3f3f3f3f);
        f[0] = 0;
        for (int t = 1; t * t <= n; t++) {
            int x = t * t;
            for (int j = x; j <= n; j++) {
                f[j] = Math.min(f[j], f[j - x] + 1);
            }
        }
        return f[n];
    }

    /**
     * 852. 山脉数组的峰顶索引
     * 符合下列属性的数组 arr 称为 山脉数组 ：
     * arr.length >= 3
     * 存在 i（0 < i < arr.length - 1）使得：
     * arr[0] < arr[1] < ... arr[i-1] < arr[i]
     * arr[i] > arr[i+1] > ... > arr[arr.length - 1]
     * 给你由整数组成的山脉数组 arr ，返回任何满足 arr[0] < arr[1] < ... arr[i - 1] < arr[i] > arr[i + 1] > ... > arr[arr.length - 1] 的下标 i
     */
    public int peakIndexInMountainArray(int[] arr) {
        //等价替换成找到数据arr中的最大值的下标
        int n = arr.length;
        int l=1;
        int r= n-1;
        while(l<r){
            int mid = l+r+1>>1;
            if(arr[mid-1]<arr[mid]){
                l=mid;
            }else {
                r=mid-1;
            }
        }
        return r;
    }
    /**
     *1449. 数位成本和为目标值的最大数字
     * 给你一个整数数组 cost 和一个整数 target 。请你返回满足如下规则可以得到的 最大 整数：
     *
     * 给当前结果添加一个数位（i + 1）的成本为 cost[i] （cost 数组下标从 0 开始）。
     * 总成本必须恰好等于 target 。
     * 添加的数位中没有数字 0 。
     * 由于答案可能会很大，请你以字符串形式返回。
     *
     * 如果按照上述要求无法得到任何整数，请你返回 "0" 。
     */
    public String largestNumber(int[] cost, int t) {
        //定义dp数组，f为当前剩余成本下可以最多可以选择几个数。
        int[] f = new int[t + 1];
        //填充数组
        Arrays.fill(f, Integer.MIN_VALUE);
        //初始化
        f[0] = 0;
        for (int i = 1; i <= 9; i++) {
            //从小到大开始遍历，u为当前数字的成本
            int u = cost[i - 1];
            //如果取用当前数字，则最多能取用多少个当前数
            for (int j = u; j <= t; j++) {
                f[j] = Math.max(f[j], f[j - u] + 1);
            }
        }
        if (f[t] < 0) return "0";
        String ans = "";
        //确定最大数后开始从大到小开始
        for (int i = 9, j = t; i >= 1; i--) {
            int u = cost[i - 1];
            //如果当前选择的数字满足f[j] == f[j - u] + 1（说明这个数字是符合取最多数的结果集里的）
            while (j >= u && f[j] == f[j - u] + 1) {
                ans += String.valueOf(i);
                j -= u;
            }
        }
        return ans;
    }

    /**
     * 877. 石子游戏
     * 亚历克斯和李用几堆石子在做游戏。偶数堆石子排成一行，每堆都有正整数颗石子 piles[i] 。
     *
     * 游戏以谁手中的石子最多来决出胜负。石子的总数是奇数，所以没有平局。
     *
     * 亚历克斯和李轮流进行，亚历克斯先开始。 每回合，玩家从行的开始或结束处取走整堆石头。 这种情况一直持续到没有更多的石子堆为止，此时手中石子最多的玩家获胜。
     *
     * 假设亚历克斯和李都发挥出最佳水平，当亚历克斯赢得比赛时返回 true ，当李赢得比赛时返回 false 。
     */
    public boolean stoneGame(int[] piles) {
        int n = piles.length;
        int[][] f= new int[n][n];
        int[][] s= new int[n][n];
        for(int j=0;j<n;j++){
            f[j][j] = piles[j];
            for(int i=j-1;i>=0;i--){
                f[i][j] = Math.max(piles[i]+s[i+1][j],piles[j]+s[i][j-1]);
                s[i][j] = Math.min(f[i+1][j],f[i][j-1]);
            }
        }
        return f[0][n-1]>s[0][n-1];
    }

    /**
     * 752. 打开转盘锁
     * 你有一个带有四个圆形拨轮的转盘锁。每个拨轮都有10个数字： '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' 。
     * 每个拨轮可以自由旋转：例如把 '9' 变为 '0'，'0' 变为 '9' 。每次旋转都只能旋转一个拨轮的一位数字。
     * 锁的初始数字为 '0000' ，一个代表四个拨轮的数字的字符串。
     * 列表 deadends 包含了一组死亡数字，一旦拨轮的数字和列表里的任何一个元素相同，这个锁将会被永久锁定，无法再被旋转。
     * 字符串 target 代表可以解锁的数字，你需要给出解锁需要的最小旋转次数，如果无论如何不能解锁，返回 -1 。
     */
    int step = 0;
    public int openLock(String[] deadends, String target) {
        if ("0000".equals(target)) {
            return 0;
        }
        HashMap<String,Boolean> map =new HashMap<>();
        for(String s:deadends){
            map.put(s,true);
        }
        if (map.containsKey("0000")) {
            return -1;
        }
        Queue<String> queue = new LinkedList<>();
        queue.offer("0000");
        Queue<String> queue2 = new LinkedList<>();
        queue.offer(target);
        HashMap<String,Integer> seed = new HashMap<>();
        HashMap<String,Integer> seed2 = new HashMap<>();
        seed.put("0000",0);
        seed2.put(target,0);
        while(!queue.isEmpty()&&!queue2.isEmpty()){
            ++step;
            Integer res ;
            if(queue.size()<=queue2.size()){
                res = process752(queue,seed,seed2,map);
            }else {
                res = process752(queue2,seed2,seed,map);
            }
            if(res!=-1){
                return res;
            }
        }
        return -1;
    }
    private Integer process752(Queue<String> queue,HashMap<String,Integer> seed,HashMap<String,Integer> seed2,HashMap<String,Boolean> map){
        int size = queue.size();
        for(int i=0;i<size;i++){
            String status = queue.poll();
            for(String nextStatus:get(status)){
                if(!seed.containsKey(nextStatus)&&!map.containsKey(nextStatus)){
                    if(seed2.containsKey(nextStatus)){
                        return seed.get(nextStatus)+seed2.get(nextStatus);
                    }else {
                        queue.offer(nextStatus);
                        seed.put(nextStatus,step);
                    }
                }
            }
        }
        return -1;
    }

    public char numPrev(char x) {
        return x == '0' ? '9' : (char) (x - 1);
    }

    public char numSucc(char x) {
        return x == '9' ? '0' : (char) (x + 1);
    }

    // 枚举 status 通过一次旋转得到的数字
    public List<String> get(String status) {
        List<String> ret = new ArrayList<String>();
        char[] array = status.toCharArray();
        for (int i = 0; i < 4; ++i) {
            char num = array[i];
            array[i] = numPrev(num);
            ret.add(new String(array));
            array[i] = numSucc(num);
            ret.add(new String(array));
            array[i] = num;
        }
        return ret;
    }

    /**
     * 909. 蛇梯棋
     */
    int n;
    int[] nums;
    public int snakesAndLadders(int[][] board) {
        n = board.length;
        if (board[0][0] != -1) return -1;
        //把数组变成一维数组
        nums = new int[n * n + 1];
        boolean isRight = true;
        for (int i = n - 1, idx = 1; i >= 0; i--) {
            for (int j = (isRight ? 0 : n - 1); isRight ? j < n : j >= 0; j += isRight ? 1 : -1) {
                nums[idx++] = board[i][j];
            }
            isRight = !isRight;
        }
        int ans = bfs();
        return ans;
    }
    int bfs() {
        Deque<Integer> d = new ArrayDeque<>();
        Map<Integer, Integer> m = new HashMap<>();
        d.addLast(1);
        m.put(1, 0);
        while (!d.isEmpty()) {
            int poll = d.pollFirst();
            int step = m.get(poll);
            if (poll == n * n) return step;
            for (int i = 1; i <= 6; i++) {
                int np = poll + i;
                if (np <= 0 || np > n * n) continue;
                if (nums[np] != -1) np = nums[np];
                if (m.containsKey(np)) continue;
                m.put(np, step + 1);
                d.addLast(np);
            }
        }
        return -1;
    }

    /**
     * 815. 公交路线
     * 给你一个数组 routes ，表示一系列公交线路，其中每个 routes[i] 表示一条公交线路，第 i 辆公交车将会在上面循环行驶。
     * 例如，路线 routes[0] = [1, 5, 7] 表示第 0 辆公交车会一直按序列 1 -> 5 -> 7 -> 1 -> 5 -> 7 -> 1 -> ... 这样的车站路线行驶。
     * 现在从 source 车站出发（初始时不在公交车上），要前往 target 车站。 期间仅可乘坐公交车。
     * 求出 最少乘坐的公交车数量 。如果不可能到达终点车站，返回 -1 。
     */
    public int numBusesToDestination(int[][] routes, int source, int target) {
        if(source==target){
            return 0;
        }
        //如果能到达目的地，每辆车最多坐一次，所以在车的角度进行广度优先遍历
        Map<Integer,List<Integer>> map = new HashMap<>();
        for(int i=0;i<routes.length;i++){
            for(int j=0;j<routes[i].length;j++){
                if(!map.containsKey(routes[i][j])){
                    List<Integer> a = new ArrayList<>();
                    a.add(i);
                    map.put(routes[i][j],a);
                }else {
                    List<Integer> a = map.get(routes[i][j]);
                    a.add(i);
                }
            }
        }
        if(!map.containsKey(source)||!map.containsKey(target)){
            return -1;
        }
        //记录当前车辆是否坐过
        Map<Integer,Integer> step = new HashMap<>();
        Queue<Integer> queue =new LinkedList<>();
        queue.add(source);
        step.put(source,0);
        while(!queue.isEmpty()){
            //获取当前节点可以乘坐的公交路线
            int now = queue.poll();
            int res = step.get(now)+1;
            List<Integer> next = map.get(now);
            for(int i=0;i<next.size();i++){
                int[] route = routes[next.get(i)];
                for(int a:route){
                    if(!step.containsKey(a)&&a!=now){
                        List<Integer> list = map.get(a);
                        list.remove(next.get(i));
                        queue.add(a);
                        step.put(a,res);
                    }
                    if(a==target){
                        return res;
                    }
                }
            }
        }
        return -1;
    }

    /**
     * 168. Excel表列名称
     * 给你一个整数 columnNumber ，返回它在 Excel 表中相对应的列名称。
     * A -> 1
     * B -> 2
     * C -> 3
     * Z -> 26
     * AA -> 27
     * AB -> 28
     */

    public String convertToTitle(int columnNumber) {
        char[] dic = new char[26];
        for(int i=0;i<dic.length;i++){
            dic[i] = (char)('A'+i);
        }
        StringBuilder sb = new StringBuilder();
        int tmp;
        while(columnNumber>0){
            columnNumber--;
            tmp = columnNumber%26;
            sb.insert(0,dic[tmp]);
            columnNumber/=26;
        }
        return sb.toString();
    }

    /**
     * LCP 07. 传递信息
     * 小朋友 A 在和 ta 的小伙伴们玩传信息游戏，游戏规则如下：
     *
     * 有 n 名玩家，所有玩家编号分别为 0 ～ n-1，其中小朋友 A 的编号为 0
     * 每个玩家都有固定的若干个可传信息的其他玩家（也可能没有）。传信息的关系是单向的（比如 A 可以向 B 传信息，但 B 不能向 A 传信息）。
     * 每轮信息必须需要传递给另一个人，且信息可重复经过同一个人
     */
    public int numWays(int n, int[][] relation, int k) {
        Map<Integer,List<Integer>> map = new HashMap<>();
        for(int i=0;i<relation.length;i++){
            if(map.containsKey(relation[i][0])){
                 map.get(relation[i][0]).add(relation[i][1]);
            }else {
                List<Integer> list= new ArrayList<>();
                list.add(relation[i][1]);
                map.put(relation[i][0],list);
            }
        }
        Queue<Integer> queue =new LinkedList<>();
        queue.add(0);
        int res = 0;
        while(!queue.isEmpty()&&k>0){
            k--;
            int size = queue.size();
            while(size-->0){
                Integer now = queue.poll();
                List<Integer> list = map.get(now);
                if(list!=null){
                    for(int i=0;i<list.size();i++){
                        queue.add(list.get(i));
                    }
                }
            }
        }
        while(!queue.isEmpty()){
            if(queue.poll()==n-1){
                res++;
            }
        }
        return res;
    }

    /**
     * 1833. 雪糕的最大数量
     * 夏日炎炎，小男孩 Tony 想买一些雪糕消消暑。
     *
     * 商店中新到 n 支雪糕，用长度为 n 的数组 costs 表示雪糕的定价，其中 costs[i] 表示第 i 支雪糕的现金价格。Tony 一共有 coins 现金可以用于消费，他想要买尽可能多的雪糕。
     *
     * 给你价格数组 costs 和现金量 coins ，请你计算并返回 Tony 用 coins 现金能够买到的雪糕的 最大数量 。
     *
     * 注意：Tony 可以按任意顺序购买雪糕。
     */
    public int maxIceCream(int[] costs, int coins) {
        Arrays.sort(costs);
        int res = 0;
        for(int i=0;i<costs.length;i++){
            if(coins>=costs[i]){
                res++;
                coins-=costs[i];
            }else {
                break;
            }
        }
        return res;
    }

    /**
     * 72. 编辑距离
     * 给你两个单词 word1 和 word2，请你计算出将 word1 转换成 word2 所使用的最少操作数 。
     *
     * 你可以对一个单词进行如下三种操作：
     *
     * 插入一个字符
     * 删除一个字符
     * 替换一个字符
     */
    public int minDistance(String word1, String word2) {
        int n = word1.length();
        int m = word2.length();
        if (n * m == 0) {
            return n + m;
        }
        int[][] dp = new int[n][m];
        // 边界状态初始化
        for (int i = 0; i < n + 1; i++) {
            dp[i][0] = i;
        }
        for (int j = 0; j < m + 1; j++) {
            dp[0][j] = j;
        }
        for (int i = 1; i < n + 1; i++) {
            for (int j = 1; j < m + 1; j++) {
                int left = dp[i - 1][j] + 1;
                int down = dp[i][j - 1] + 1;
                int left_down = dp[i - 1][j - 1];
                if (word1.charAt(i - 1) != word2.charAt(j - 1)) {
                    left_down += 1;
                }
                dp[i][j] = Math.min(left, Math.min(down, left_down));
            }
        }
        return dp[n][m];
    }

    /**
     * 645. 错误的集合
     * 集合 s 包含从 1 到 n 的整数。不幸的是，因为数据错误，导致集合里面某一个数字复制了成了集合里面的另外一个数字的值，导致集合 丢失了一个数字 并且 有一个数字重复 。
     *
     * 给定一个数组 nums 代表了集合 S 发生错误后的结果。
     *
     * 请你找出重复出现的整数，再找到丢失的整数，将它们以数组的形式返回。
     */
    public int[] findErrorNums(int[] nums) {
        int n = nums.length;
        int[] cnts = new int[n + 1];
        for (int x : nums) cnts[x]++;
        int[] ans = new int[2];
        for (int i = 1; i <= n; i++) {
            if (cnts[i] == 0) ans[1] = i;
            if (cnts[i] == 2) ans[0] = i;
        }
        return ans;
    }


    /**
     * 726. 原子的数量
     * 给定一个化学式formula（作为字符串），返回每种原子的数量。
     * 原子总是以一个大写字母开始，接着跟随0个或任意个小写字母，表示原子的名字。
     * 如果数量大于 1，原子后会跟着数字表示原子的数量。如果数量等于 1 则不会跟数字。例如，H2O 和 H2O2 是可行的，但 H1O2 这个表达是不可行的。
     * 两个化学式连在一起是新的化学式。例如 H2O2He3Mg4 也是化学式。
     * 一个括号中的化学式和数字（可选择性添加）也是化学式。例如 (H2O2) 和 (H2O2)3 是化学式。
     */


    public String countOfAtoms(String formula) {
            return "123";
    }

    /**
     * 1418. 点菜展示表
     * 给你一个数组 orders，表示客户在餐厅中完成的订单，确切地说， orders[i]=[customerNamei,tableNumberi,foodItemi] ，
     * 其中 customerNamei 是客户的姓名，tableNumberi 是客户所在餐桌的桌号，而 foodItemi 是客户点的餐品名称。
     * 请你返回该餐厅的 点菜展示表 。在这张表中，表中第一行为标题，其第一列为餐桌桌号 “Table” ，后面每一列都是按字母顺序排列的餐品名称。
     * 接下来每一行中的项则表示每张餐桌订购的相应餐品数量，第一列应当填对应的桌号，后面依次填写下单的餐品数量。
     */
    public List<List<String>> displayTable(List<List<String>> orders) {
        //记录每张桌子所有的食物清单
        Map<String,Map<String,Integer>> map = new HashMap<>();
        //记录全部的食物
        Set<String> foods = new TreeSet<>();
        //记录全部的桌子
        Set<Integer> tables = new TreeSet<>();
        for(int i=0;i<orders.size();i++){
            List<String> order = orders.get(i);
            String table = order.get(1);
            String food = order.get(2);
            foods.add(food);
            tables.add(Integer.valueOf(table));
            if(map.containsKey(table)){
                Map<String, Integer> tmp = map.get(table);
                if(tmp.containsKey(food)){
                   tmp.put(food,tmp.get(food)+1);
                }else {
                    tmp.put(food,1);
                }
            }else {
                Map<String, Integer> tmp =new HashMap<>();
                tmp.put(food,1);
                map.put(table,tmp);
            }
        }
        List<List<String>> res = new ArrayList<>();
        List<String> title = new ArrayList<>();
        title.add("Table");
        title.addAll(foods);
        res.add(title);
        for(Integer t:tables){
            Map<String, Integer> tmp = map.get(t.toString());
            List<String> list = groupFood(tmp,foods,t.toString());
            res.add(list);
        }
        return res;
    }

    private List<String> groupFood(Map<String, Integer> tmp,Set<String> foods,String table){
        List<String> res = new ArrayList<>();
        res.add(table);
        for(String f:foods){
            if(tmp.containsKey(f)){
                res.add(tmp.get(f).toString());
            }else {
                res.add("0");
            }
        }
        return res;
    }

    /**
     * 1711. 大餐计数
     * 大餐 是指 恰好包含两道不同餐品 的一餐，其美味程度之和等于 2 的幂。
     * 你可以搭配 任意 两道餐品做一顿大餐。
     * 给你一个整数数组 deliciousness ，其中 deliciousness[i] 是第 i道餐品的美味程度，返回你可以用数组中的餐品做出的不同 大餐 的数量。
     * 结果需要对 10^9+ 7 取余。
     * 注意，只要餐品下标不同，就可以认为是不同的餐品，即便它们的美味程度相同。
     */
    public int countPairs(int[] deliciousness) {
        Long res = 0L;
        int mod = 1000000007;
        int n = deliciousness.length;
        List<Integer> list =new ArrayList<>();
        int a = 1;
        list.add(a);
        for(int i=0;i<21;i++){
            a*=2;
            list.add(a);
        }
        Map<Integer,Integer> map =new HashMap<>();
        for(int i=0;i<n;i++){
            if(map.containsKey(deliciousness[i])){
                map.put(deliciousness[i],map.get(deliciousness[i])+1);
            }else {
                map.put(deliciousness[i],1);
            }
        }
        for(Integer b:map.keySet()){
            for(Integer c:list){
                int d = c-b;
                if(map.containsKey(d)){
                    if(d==b){
                        res+=map.get(d)*(map.get(b)-1);
                    }else {
                        res+=map.get(d)*map.get(b);
                    }
                }
            }
        }
        return (int)(res%mod);
    }

    /**
     * 930. 和相同的二元子数组
     * 给你一个二元数组 nums ，和一个整数 goal ，请你统计并返回有多少个和为 goal 的 非空 子数组。
     *
     * 子数组 是数组的一段连续部分。
     */
    public int numSubarraysWithSum(int[] nums, int goal) {
        int n = nums.length;
        HashMap<Integer,Integer> map = new HashMap<>();
        int[] sums = new int[n+1];
        for(int i=1;i<=n;i++){
            sums[i]=sums[i-1]+nums[i-1];
        }
        map.put(0,1);
        int res = 0;
        for(int i=0;i<n;i++){
            int r = sums[i+1];
            int l = r-goal;
            res += map.getOrDefault(l,0);
            map.put(r,map.getOrDefault(r,0)+1);
        }
        return res;
    }

    /**
     * 面试题 17.10. 主要元素
     * 数组中占比超过一半的元素称之为主要元素。给你一个 整数 数组，找出其中的主要元素。若没有，返回 -1 。请设计时间复杂度为 O(N) 、空间复杂度为 O(1) 的解决方案。
     */
    public int majorityElement(int[] nums) {
        int count =1;
        int res = nums[0];
        for(int i =1;i<nums.length;i++){
            if(count==0){
                res = nums[i];
                count++;
            }else {
                if(res==nums[i]){
                    count++;
                }else {
                    count--;
                }
            }
        }
        int half = 0;
        for(int i=0;i<nums.length;i++){
            if(nums[i]==res){
                half++;
            }
        }
        if(half>nums.length/2){
            return res;
        }else {
            return -1;
        }
    }

    /**
     * 981. 基于时间的键值存储
     * 创建一个基于时间的键值存储类 TimeMap，它支持下面两个操作：
     *
     * 1. set(string key, string value, int timestamp)
     *
     * 存储键 key、值 value，以及给定的时间戳 timestamp。
     * 2. get(string key, int timestamp)
     *
     * 返回先前调用 set(key, value, timestamp_prev) 所存储的值，其中 timestamp_prev <= timestamp。
     * 如果有多个这样的值，则返回对应最大的  timestamp_prev 的那个值。
     * 如果没有值，则返回空字符串（""）。
     */
    class TimeMap {

        HashMap<String,TreeMap<Integer,String>> map;

        /** Initialize your data structure here. */
        public TimeMap() {
            this.map= new HashMap<>();

        }

        public void set(String key, String value, int timestamp) {
            if(map.containsKey(key)){
                TreeMap<Integer,String> timeMap = map.get(key);
                timeMap.put(timestamp,value);
            }else {
                TreeMap<Integer,String> timeMap = new TreeMap<>();
                timeMap.put(timestamp,value);
                map.put(key,timeMap);
            }
        }

        public String get(String key, int timestamp) {
            Map.Entry<Integer, String> entry = map.getOrDefault(key, new TreeMap<>()).floorEntry(timestamp);
            return entry == null ? "" : entry.getValue();
        }
    }


    /**
     * 274. H 指数
     * 给定一位研究者论文被引用次数的数组（被引用次数是非负整数）。编写一个方法，计算出研究者的 h 指数。
     *
     * h 指数的定义：h 代表“高引用次数”（high citations），一名科研人员的 h 指数是指他（她）的 （N 篇论文中）总共有 h 篇论文分别被引用了至少 h 次。且其余的 N - h 篇论文每篇被引用次数 不超过 h 次。
     */
    public int hIndex(int[] citations) {
        Arrays.sort(citations);
        int h = 0, i = citations.length - 1;
        while (i >= 0 && citations[i] > h) {
            h++;
            i--;
        }
        return h;
    }

    /**
     * 275. H 指数 II
     * 给定一位研究者论文被引用次数的数组（被引用次数是非负整数），数组已经按照 升序排列 。编写一个方法，计算出研究者的 h 指数。
     *
     * h 指数的定义: “h 代表“高引用次数”（high citations），一名科研人员的 h 指数是指他（她）的 （N 篇论文中）总共有 h 篇论文分别被引用了至少 h 次。（其余的 N - h 篇论文每篇被引用次数不多于 h 次。）"
     */
    public int hIndex2(int[] citations) {
        int n = citations.length;
        int left = 0;
        int right = n-1;
        while(left<right){
            int mid = (left+right)/2;
            int h = n-mid;
            if(citations[mid]>=h){
                right = mid;
            }else {
                left = mid+1;
            }
        }
        return citations[right]>=n-right?n-right:0;
    }


    /**
     * 有一个字符串，如果同一个字符重复出现3次及以上，则让它消除，给一个字符串，输出消除后的字符串。
     */

    public String xxl(String s){
        char[] c = s.toCharArray();
        int n = c.length;
        int i=0;
        Stack<Character> stack =new Stack<>();
        while(i<n){
            int tmp=i;
            if(stack.isEmpty()){
                stack.push(c[i]);
            }else {
                if(stack.peek()==c[tmp]){
                    while(stack.peek()==c[tmp]){
                        stack.push(c[tmp++]);
                    }
                    if(tmp-i>=2){
                        int t = tmp;
                        while(tmp>=i){
                            stack.pop();
                            tmp--;
                        }
                        i=t-1;
                    }else if(stack.size()>2){
                        char a = stack.pop();
                        char b = stack.pop();
                        char d = stack.pop();
                        if(a==b&&b==d){

                        }else {
                            stack.push(d);
                            stack.push(b);
                            stack.push(a);
                        }
                    }
                }else {
                    stack.push(c[i]);
                }
            }
            i++;
        }
        StringBuilder sb = new StringBuilder();
        while(!stack.isEmpty()){
            sb.insert(0,stack.pop());
        }
        return sb.toString();
    }

    /**
     * 面试题 10.02. 变位词组
     * 编写一种方法，对字符串数组进行排序，将所有变位词组合在一起。变位词是指字母相同，但排列不同的字符串。
     */
    public List<List<String>> groupAnagrams(String[] ss) {
        List<List<String>> ans = new ArrayList<>();
        Map<String, List<String>> map = new HashMap<>();
        for (String s : ss) {
            char[] cs = s.toCharArray();
            //对字符数组进行排序
            Arrays.sort(cs);
            String key = String.valueOf(cs);
            List<String> list = map.getOrDefault(key, new ArrayList<>());
            list.add(s);
            map.put(key, list);
        }
        for (String key : map.keySet()) ans.add(map.get(key));
        return ans;
    }

    /**
     * 1838. 最高频元素的频数
     * 元素的 频数 是该元素在一个数组中出现的次数。
     *
     * 给你一个整数数组 nums 和一个整数 k 。在一步操作中，你可以选择 nums 的一个下标，并将该下标对应元素的值增加 1 。
     *
     * 执行最多 k 次操作后，返回数组中最高频元素的 最大可能频数 。
     */
}

