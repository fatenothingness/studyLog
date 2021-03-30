import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import java.util.Stack;

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

}
