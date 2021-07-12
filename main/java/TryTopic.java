public class TryTopic {

    public static void main(String args[]){
        OneTopicEveryday o = new OneTopicEveryday();
        int[] a = new int[]{1,0,1,0,1};
        char[] c = new char[]{};
        int[][] a2 = new int[][]{};
        char[][] c2 = new char[][]{};
        System.out.println(o.numSubarraysWithSum(a,2));
    }
}
