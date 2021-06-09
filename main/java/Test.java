import com.alibaba.fastjson.JSONObject;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class Test {


    public static void main(String args[]){
        OneTopicEveryday oneTopicEveryday = new OneTopicEveryday();
        int[] candiesCount = new int[]{13,21,34,55,89,14,23,37,61,98};
        int[][] matrix = new int[][]{{3,1,2},{4,10,3},{3,10,100},{4,100,30},{1,3,1}};
        int booleans = oneTopicEveryday.lastStoneWeightII(candiesCount);
        System.out.println(booleans);
    }
}
