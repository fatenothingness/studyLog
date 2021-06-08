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
     * @param target
     * @return
     */
    public boolean search(int[] nums, int target) {
        int k = -1;
        int len = nums.length;
        //先判断旋转后短点的位置，如果找不到则说明数组的全部数都相同
        for(int i=1;i<len;i++){
            if(nums[i]<nums[i-1]){
                k = i;
            }
        }
        if(k==-1){
            return target==nums[0];
        }
        //在断点的左边和右边都是非降序数组，左边是0 k-1 右边是k len-1 且 nums[0]>=nums[len-1]
        if(target>nums[k-1]||target<nums[k]){
            return false;
        }
        int mid;
        int left;
        int right;
        //在左边
        if(target>nums[len-1]){
            left = 0;
            right = k-1;
            mid = left+right+1/2;
            while(left<right){

            }
        }
        //在右边
        else if(target<nums[len-1]){

        }else {
            return true;
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
     * 给定一个无序数组，找出其中第k大的数
     */

}
