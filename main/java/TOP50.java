import java.util.*;

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
     * 4. 寻找两个正序数组的中位数
     * 给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。
     */
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int len = nums1.length+nums2.length;
        int mid;
        boolean flag = false;
        if(len%2==0){
            mid = (len/2)+1;
            flag = true;
        }else{
            mid = (len+1)/2;
        }
        int[] tmp = new int[mid];
        int a = 0;
        int b = 0;
        int i = 0;
        while(i<mid&&a<nums1.length&&b<nums2.length){
            if(nums1[a]<nums2[b]){
                tmp[i++] = nums1[a++];
            }else{
                tmp[i++] = nums2[b++];
            }
        }
        if(i!=mid){
            if(a!=nums1.length){
                while(i<mid){
                    tmp[i++] = nums1[a++];
                }
            }else{
                while(i<mid){
                    tmp[i++] = nums2[b++];
                }
            }
        }
        if(flag){
            return (tmp[mid-1]+tmp[mid-2])/2.0;
        }
        else{
            return tmp[mid-1];
        }
    }

    /**
     * 5. 最长回文子串
     * 给你一个字符串 s，找到 s 中最长的回文子串。
     * 方法：动态规划，定义dp数组 boolean[i][j] 为：在i到是否为回文串
     */
    public String longestPalindrome(String s){
        int n = s.length();
        if(n<2){
            return s;
        }
        int left = 0;
        int right = 0;
        int max = 0;
        boolean[][] dp = new boolean[n][n];
        for(int i=0;i<n;i++){
            dp[i][i] = true;
        }
        for(int j=1;j<n;j++){
            for(int i=0;i<j;i++){
                int v = j-i+1;
                if(s.charAt(i)==s.charAt(j)&&(j-i<=2||dp[i+1][j-1])){
                    dp[i][j]=true;
                    if(v>max){
                        left=i;
                        right=j;
                        max=v;
                    }
                }
            }
        }
        return s.substring(left,right+1);
    }

    /**
     * 49. 字母异位词分组
     * 给定一个字符串数组，将字母异位词组合在一起。字母异位词指字母相同，但排列不同的字符串。
     */
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> map = new HashMap<String, List<String>>();
        for (String str : strs) {
            char[] array = str.toCharArray();
            Arrays.sort(array);
            String key = new String(array);
            List<String> list = map.getOrDefault(key, new ArrayList<String>());
            list.add(str);
            map.put(key, list);
        }
        return new ArrayList<List<String>>(map.values());
    }
}
