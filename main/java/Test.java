import com.alibaba.fastjson.JSONObject;

import java.util.Map;

public class Test {


    public static void main(String args[]){
        OneTopicEveryday oneTopicEveryday = new OneTopicEveryday();
        ListNode head = new ListNode(1,new ListNode(1,new ListNode(1,new ListNode(5,new ListNode(5,null)))));
        ListNode w = new ListNode(2,new ListNode(1,new ListNode(1,new ListNode(5,new ListNode(5,null)))));
        ListNode a = oneTopicEveryday.deleteDuplicates1(head);
        SortUtil sortUtil = new SortUtil();
        int[] matrix = new int[]{111311, 1113};
        int[] arr = new int[]{1,2,3,4,5,6,7};
        int x = oneTopicEveryday.find(arr,2);
        System.out.println(x);
    }
}
