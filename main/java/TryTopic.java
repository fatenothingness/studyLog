import Topic.JulyTopic;
import Topic.OctTopic;
import Topic.SeptemberTopic;
import util.ListNode;
import util.TreeNode;

import java.io.IOException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

public class TryTopic {

    private static final String TIME="yyyyMMddHHmmssSSS";

    public int add(int a,int b){
        return a+b;
    }

    public static void main(String args[]) throws IOException, ParseException {
        SeptemberTopic s = new SeptemberTopic();
        OctTopic o = new OctTopic();
        JulyTopic j = new JulyTopic();
        int[] a = new int[]{1,1,1,2};
        int[] b = new int[]{0,1,2};
        char[] c = new char[]{};
        int[][] a2 = new int[][]{{1,2},{3},{3},{}};
        char[][] c2 = new char[][]{};
        String[] word = new String[]{"ask","not","what","your","country","can","do","for","you","ask","what","you","can","do","for","your","country"};
        TreeNode q = new TreeNode(1,new TreeNode(2,new TreeNode(3),new TreeNode(4,new TreeNode(5),new TreeNode(6))),new TreeNode(7,new TreeNode(8),new TreeNode(9)));
        ListNode l = new ListNode(1,new ListNode(2,new ListNode(3,new ListNode(4,new ListNode(5,new ListNode(6,new ListNode(7,null)))))));
        System.out.println();
    }
}
