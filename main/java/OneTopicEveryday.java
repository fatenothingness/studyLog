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
}
