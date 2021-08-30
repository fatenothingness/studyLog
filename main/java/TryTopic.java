import org.checkerframework.checker.units.qual.A;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

public class TryTopic {

    public int add(int a,int b){
        return a+b;
    }

    public static void main(String args[]) throws IOException {
        TryTopic o = new TryTopic();
        JulyTopic j = new JulyTopic();
        int[] a = new int[]{3,5,3,4};
        int[] b = new int[]{9,3,5,1,7,4};
        char[] c = new char[]{};
        int[][] a2 = new int[][]{{1,2},{3},{3},{}};
        char[][] c2 = new char[][]{};
        TreeNode q = new TreeNode(3,new TreeNode(5,new TreeNode(6),new TreeNode(2,new TreeNode(7),new TreeNode(4))),new TreeNode(1,new TreeNode(0),new TreeNode(8)));
        System.out.println(j.numRescueBoats(a,5));
    }
}
