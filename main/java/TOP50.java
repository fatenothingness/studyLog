import java.util.HashMap;
import java.util.Map;

public class TOP50 {


    /**
     * 3. 无重复字符的最长子串
     * 给定一个字符串，请你找出其中不含有重复字符的 最长子串 的长度。
     * 方法：滑动窗口
     */
    public int lengthOfLongestSubstring(String s) {
        char[] chars = s.toCharArray();
        Map<Character,Integer> map =new HashMap<>();
        int i=0;
        int left = 0;
        int res = 0;
        while(i<chars.length){
            //如果当前数已经存在，窗口向左滑动
            if(map.containsKey(chars[i])){
                map.remove(chars[i]);
                left++;
            }else {
                //不存在这窗口向右滑，记录窗口长度的最大值
                map.put(chars[i],1);
                i++;
                res = Math.max(res,i-left+1);
            }
        }
        return res;
    }

    /**
     * 5. 最长回文子串
     * 给你一个字符串 s，找到 s 中最长的回文子串。
     * 方法：动态规划，定义dp数组 int[i][j] 为：在i到j回文串的长度
     */
}
