import com.alibaba.fastjson.JSONObject;

import java.util.Map;

public class Test {


    public static void main(String args[]){
        OneTopicEveryday oneTopicEveryday = new OneTopicEveryday();
        ListNode head = new ListNode(1,new ListNode(1,new ListNode(1,new ListNode(5,new ListNode(5,null)))));
        ListNode a = oneTopicEveryday.deleteDuplicates1(head);
        int[] matrix = new int[]{1,2};
        int  f = oneTopicEveryday.removeDuplicates(matrix);
        System.out.println(f);
    }
}
