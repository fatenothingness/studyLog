public class Test {


    public static void main(String args[]){
        OneTopicEveryday oneTopicEveryday = new OneTopicEveryday();
        ListNode head = new ListNode(1,new ListNode(2,new ListNode(3,new ListNode(4,new ListNode(5,null)))));
        ListNode i = oneTopicEveryday.reverseBetween(head,5,5);
        System.out.println(i);
    }
}
