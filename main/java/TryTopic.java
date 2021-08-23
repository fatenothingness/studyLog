public class TryTopic {

    public static void main(String args[]){
        OneTopicEveryday o = new OneTopicEveryday();
        JulyTopic j = new JulyTopic();
        int[] a = new int[]{1,2,3,4,5,6,8,10};
        int[] b = new int[]{9,3,5,1,7,4};
        char[] c = new char[]{'a','a','b','b','c','c','c'};
        int[][] a2 = new int[][]{};
        char[][] c2 = new char[][]{};
        TreeNode q = new TreeNode(3,new TreeNode(5,new TreeNode(6),new TreeNode(2,new TreeNode(7),new TreeNode(4))),new TreeNode(1,new TreeNode(0),new TreeNode(8)));
        System.out.println(j.compress(c));
    }
}
