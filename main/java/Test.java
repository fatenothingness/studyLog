import com.alibaba.fastjson.JSONObject;
import org.checkerframework.checker.units.qual.A;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

public class Test {


    public static void main(String args[]){
        OneTopicEveryday oneTopicEveryday = new OneTopicEveryday();
        SwordToOffer swordToOffer = new SwordToOffer();
        SortUtil sortUtil = new SortUtil();
        int[] candiesCount = new int[]{13,21,34,55,89,14,23,37,61,98};
        int[][] matrix = new int[][]{{1,4,7,11,15},{2,5,8,12,19},{3,6,9,16,22}};
        int booleans = oneTopicEveryday.lastStoneWeightII(candiesCount);
        sortUtil.mergeSort(candiesCount);
        System.out.println(swordToOffer.movingCount(38,15,9));
    }

}

