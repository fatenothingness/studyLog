import com.alibaba.fastjson.JSONObject;

import java.util.Map;

public class Test {


    public static void main(String args[]){
        OneTopicEveryday oneTopicEveryday = new OneTopicEveryday();
        ListNode head = new ListNode(1,new ListNode(1,new ListNode(1,new ListNode(5,new ListNode(5,null)))));
        ListNode a = oneTopicEveryday.deleteDuplicates1(head);
        int[][] matrix = new int[][]{{1,3,5,7},{10,11,16,20},{23,30,34,60}};
        boolean  f = oneTopicEveryday.searchMatrix(matrix,20);
        System.out.println(f);
    }
}
