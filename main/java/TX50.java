public class TX50 {
    /**
     * 2. 两数相加
     * 给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。
     *
     * 请你将两个数相加，并以相同形式返回一个表示和的链表。
     *
     * 你可以假设除了数字 0 之外，这两个数都不会以 0 开头。
     */
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode a = l1;
        ListNode b = l2;
        ListNode result = new ListNode(0);
        ListNode c = result;
        boolean flag = false;
        //当两条链表都不为空时
        while(a!=null&&b!=null){
            //判断上一个位数是否有进位
            if(flag){
                if(a.val+b.val>=9){
                    result.next = new ListNode(a.val+b.val-9);
                    flag = true;
                }else{
                    result.next = new ListNode(a.val+b.val+1);
                    flag = false;
                }
            }
            else{
                if(a.val+b.val>=10){
                    result.next = new ListNode(a.val+b.val-10);
                    flag = true;
                }else{
                    result.next = new ListNode(a.val+b.val);
                    flag = false;
                }
            }
            a = a.next;
            b = b.next;
            result = result.next;
        }
        //如果都到了结尾，但是有进位，这在最后加一位 1
        if(a==null&&b==null&&flag){
            result.next = new ListNode(1);
        }
        if(a==null&&b==null){
            return c.next;
        }
        ListNode t;
        if(a==null){
            t = b;
        }
        else{
            t = a;
        }
        while(t!=null){
            if(flag){
                if(t.val+1==10){
                    result.next = new ListNode(0);
                    flag = true;
                }
                else{
                    result.next = new ListNode(t.val+1);
                    flag = false;
                }
            }
            else{
                result.next = new ListNode(t.val);
            }
            t = t.next;
            result = result.next;
            if(t==null&&flag){
                result.next = new ListNode(1);
            }
        }
        return c.next;
    }

    /**
     * 4. 寻找两个正序数组的中位数
     * 给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。
     */
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int len = nums1.length+nums2.length;
        int mid;
        boolean flag = false;
        //判断数组时奇数还是偶数，奇数时取中间值，偶数取中俩值的平均值。
        if(len%2==0){
            mid = (len/2)+1;
            flag = true;
        }else{
            mid = (len+1)/2;
        }
        //用临时数组存储前半的数组。
        int[] tmp = new int[mid];
        int a = 0;
        int b = 0;
        int i = 0;
        //双指针遍历数组，每次把最小值加入到临时数组中
        while(i<mid&&a<nums1.length&&b<nums2.length){
            if(nums1[a]<nums2[b]){
                tmp[i] = nums1[a];
                a++;
                i++;
            }else{
                tmp[i] = nums2[b];
                b++;
                i++;
            }
        }
        //判断当其中一个数组走完时，是否到整体到一半。
        if(i!=mid){
            if(a!=nums1.length){
                while(i<mid){
                    tmp[i] = nums1[a];
                    a++;
                    i++;
                }
            }else{
                while(i<mid){
                    tmp[i] = nums2[b];
                    b++;
                    i++;
                }
            }
        }
        //根据奇偶数来取中间值
        if(flag){
            return (tmp[mid-1]+tmp[mid-2])/2.0;
        }
        else{
            return tmp[mid-1];
        }
    }

    /**
     * 5. 最长回文子串
     * 给你一个字符串 s，找到 s 中最长的回文子串。
     * 思路：适用动态规划做法，每个单独的字符都是最小的回文串，定义递归数组 boolean[i][j] dp  dp[i][j] 为在i-j范围内 是否是回文串，这取决与
     * dp[i+1][j] 和 dp[i][j-1] 和 dp[i+1][j-1]
     */
    public String longestPalindrome(String s){
        int n = s.length();
        if(n<2){
            return s;
        }
        int left = 0;
        int right = 0;
        int max = 0;
        boolean[][] dp = new boolean[n][n];
        for(int i=0;i<n;i++){
            dp[i][i] = true;
        }
        for(int j=1;j<n;j++){
            for(int i=0;i<j;i++){
                int v = j-i+1;
                if(s.charAt(i)==s.charAt(j)&&(j-i<=2||dp[i+1][j-1])){
                    dp[i][j]=true;
                    if(v>max){
                        left=i;
                        right=j;
                        max=v;
                    }
                }
            }
        }
        return s.substring(left,right+1);
    }
}
