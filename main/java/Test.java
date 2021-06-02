import com.alibaba.fastjson.JSONObject;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class Test {


    public static void main(String args[]){
        OneTopicEveryday oneTopicEveryday = new OneTopicEveryday();
        int[] candiesCount = new int[]{5,2,6,4,1};
        int[][] matrix = new int[][]{{3,1,2},{4,10,3},{3,10,100},{4,100,30},{1,3,1}};
        boolean[] booleans = oneTopicEveryday.canEat(candiesCount, matrix);
        System.out.println(booleans);
    }
}
