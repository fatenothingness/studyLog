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
    public void dp(TreeNode root,List<Integer> list){
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

    public void leafDp(TreeNode root,List<Integer> list){
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
     * 给定一个无序数组，找出其中第k大的数
     */

}
