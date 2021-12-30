package Topic;

import java.util.*;

public class DecTopic {

    public static void main(String args[]){
        DecTopic d = new DecTopic();
        String s = "cc";
        int[]score = new int[]{1,1,2,2,3,3};
        String[] words = new String[]{""};
        d.isNStraightHand(score,2);
    }


    /**
     * 1446. 连续字符
     * 给你一个字符串 s ，字符串的「能量」定义为：只包含一种字符的最长非空子字符串的长度。
     *
     * 请你返回字符串的能量。
     */
    public int maxPower(String s) {
        int max = 1;
        int t = 1;
        for(int i=0;i<s.length()-1;i++){
            char l = s.charAt(i);
            char r = s.charAt(i+1);
            if(l==r){
                t++;
            }else {
                max = Math.max(max,t);
                t = 1;
            }
        }
        max = Math.max(max,t);
        return max;
    }

    /**
     * 506. 相对名次
     * 给你一个长度为 n 的整数数组 score ，其中 score[i] 是第 i 位运动员在比赛中的得分。所有得分都 互不相同 。
     *
     * 运动员将根据得分 决定名次 ，其中名次第 1 的运动员得分最高，名次第 2 的运动员得分第 2 高，依此类推。运动员的名次决定了他们的获奖情况：
     *
     * 名次第 1 的运动员获金牌 "Gold Medal" 。
     * 名次第 2 的运动员获银牌 "Silver Medal" 。
     * 名次第 3 的运动员获铜牌 "Bronze Medal" 。
     * 从名次第 4 到第 n 的运动员，只能获得他们的名次编号（即，名次第 x 的运动员获得编号 "x"）。
     * 使用长度为 n 的数组 answer 返回获奖，其中 answer[i] 是第 i 位运动员的获奖情况。
     */
    String[] ss = new String[]{"Gold Medal", "Silver Medal", "Bronze Medal"};
    public String[] findRelativeRanks(int[] score) {
        int n = score.length;
        String[] ans = new String[n];
        int[] clone = score.clone();
        Arrays.sort(clone);
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = n - 1; i >= 0; i--) map.put(clone[i], n - 1 - i);
        for (int i = 0; i < n; i++) {
            int rank = map.get(score[i]);
            ans[i] = rank < 3 ? ss[rank] : String.valueOf(rank + 1);
        }
        return ans;
    }


    /**
     * 748. 最短补全词
     * 给你一个字符串 licensePlate 和一个字符串数组 words ，请你找出并返回 words 中的 最短补全词 。
     *
     * 补全词 是一个包含 licensePlate 中所有的字母的单词。在所有补全词中，最短的那个就是 最短补全词 。
     *
     * 在匹配 licensePlate 中的字母时：
     *
     * 忽略 licensePlate 中的 数字和空格 。
     * 不区分大小写。
     * 如果某个字母在 licensePlate 中出现不止一次，那么该字母在补全词中的出现次数应当一致或者更多。
     * 例如：licensePlate = "aBc 12c"，那么它的补全词应当包含字母 'a'、'b' （忽略大写）和两个 'c' 。可能的 补全词 有 "abccdef"、"caaacab" 以及 "cbca" 。
     *
     * 请你找出并返回 words 中的 最短补全词 。题目数据保证一定存在一个最短补全词。当有多个单词都符合最短补全词的匹配条件时取 words 中 最靠前的 那个。
     * @param licensePlate
     * @param words
     * @return
     */
    public String shortestCompletingWord(String licensePlate, String[] words) {
        int[] cnt = getCnt(licensePlate);
        String ans = null;
        for (String s : words) {
            int[] cur = getCnt(s);
            boolean ok = true;
            for (int i = 0; i < 26 && ok; i++) {
                if (cnt[i] > cur[i]) ok = false;
            }
            if (ok && (ans == null || ans.length() > s.length())) ans = s;
        }
        return ans;
    }
    int[] getCnt(String s) {
        int[] cnt = new int[26];
        for (char c : s.toCharArray()) {
            if (Character.isLetter(c)) cnt[Character.toLowerCase(c) - 'a']++;
        }
        return cnt;
    }


    /**
     * 1154. 一年中的第几天
     * 给你一个字符串 date ，按 YYYY-MM-DD 格式表示一个 现行公元纪年法 日期。请你计算并返回该日期是当年的第几天。
     *
     * 通常情况下，我们认为 1 月 1 日是每年的第 1 天，1 月 2 日是每年的第 2 天，依此类推。每个月的天数与现行公元纪年法（格里高利历）一致。
     */
    public int dayOfYear(String date) {
        String[] s = date.split("-");
        int y = Integer.valueOf(s[0]);
        int m = Integer.valueOf(s[1]);
        int d = Integer.valueOf(s[2]);
        boolean r = y%4==0&&y!=1900;
        int[] month = new int[]{31,28,31,30,31,30,31,31,30,31,30,31};
        int res = 0;
        for(int i=1;i<m;i++){
            res+=month[i-1];
        }
        res+=d;
        if(m>2&&r){
            res++;
        }
        return res;
    }

    /**
     * 社交媒体网站上有 n 个用户。给你一个整数数组 ages ，其中 ages[i] 是第 i 个用户的年龄。
     *
     * 如果下述任意一个条件为真，那么用户 x 将不会向用户 y（x != y）发送好友请求：
     *
     * age[y] <= 0.5 * age[x] + 7
     * age[y] > age[x]
     * age[y] > 100 && age[x] < 100
     * 否则，x 将会向 y 发送一条好友请求。
     *
     * 注意，如果 x 向 y 发送一条好友请求，y 不必也向 x 发送一条好友请求。另外，用户不会向自己发送好友请求。
     *
     * 返回在该社交媒体网站上产生的好友请求总数。

     */

    public int numFriendRequests(int[] ages) {
        Arrays.sort(ages);
        int n = ages.length, ans = 0;
        for (int k = 0, i = 0, j = 0; k < n; k++) {
            while (i < k && !check(ages[i], ages[k])) i++;
            if (j < k) j = k;
            while (j < n && check(ages[j], ages[k])) j++;
            if (j > i) ans += j - i - 1;
        }
        return ans;
    }
    boolean check(int x, int y) {
        if (y <= 0.5 * x + 7) return false;
        if (y > x) return false;
        if (y > 100 && x < 100) return false;
        return true;
    }


    /**
     * 472. 连接词
     * 给你一个 不含重复 单词的字符串数组 words ，请你找出并返回 words 中的所有 连接词 。
     *
     * 连接词 定义为：一个完全由给定数组中的至少两个较短单词组成的字符串。
     */
    public List<String> findAllConcatenatedWordsInADict(String[] words) {
        HashMap<Character,ArrayList<String>> map = new HashMap<>();
        for(String s:words){
            if(s.equals("")){
                continue;
            }
            char f = s.charAt(0);
            if(map.containsKey(f)){
                map.get(f).add(s);
            }else {
                ArrayList<String> list = new ArrayList<>();
                list.add(s);
                map.put(f,list);
            }
        }
        List<String> res = new ArrayList<>();
        for(String s:words){
            this.check(map,s,s,0,res);
        }
        return res;
    }
    private void check(HashMap<Character,ArrayList<String>> map,String s,String x,Integer sum,List<String> res){
        if(s.equals("")){
            return;
        }
        if(x.equals("")&&sum>1){
           res.add(s);
           return;
        }
        char f = x.charAt(0);
        ArrayList<String> list = map.get(f);
        if(list==null||list.isEmpty()){
            return;
        }
        for(int i=0;i<list.size();i++){
            String t = list.get(i);
            if(t.length()>s.length()){
                continue;
            }
            if(x.startsWith(t)){
                if(x.equals(t)&&sum<1){
                    continue;
                }
                check(map,s,x.substring(t.length()),sum+1,res);
            }
        }
    }

    /**
     * 846. 一手顺子
     * Alice 手中有一把牌，她想要重新排列这些牌，分成若干组，使每一组的牌数都是 groupSize ，并且由 groupSize 张连续的牌组成。
     *
     * 给你一个整数数组 hand 其中 hand[i] 是写在第 i 张牌，和一个整数 groupSize 。如果她可能重新排列这些牌，返回 true ；否则，返回 false 。
     */
    public boolean isNStraightHand(int[] hand, int groupSize) {
        //思路：先判断hand长度是否是groupSize的整数倍，否的话返回false，
        //然后把hand进行排序，用treeMap存储数据，key为数字大小，value为数字出现的次数。
        if(hand.length%groupSize!=0){
            return false;
        }
        TreeMap<Integer,Integer> map = new TreeMap<>();
        for(int i=0;i<hand.length;i++){
            if(map.containsKey(hand[i])){
                map.put(hand[i],map.get(hand[i])+1);
            }else {
                map.put(hand[i],1);
            }
        }
        int i=0;
        int [] a = new int[hand.length];
        while(i<hand.length){
            int j = 0;
            Iterator<Map.Entry<Integer, Integer>> it = map.entrySet().iterator();
            while(j++<groupSize) {
                if(!it.hasNext()){
                    it =  map.entrySet().iterator();
                }
                Map.Entry<Integer, Integer> entry = it.next();
                a[i] = entry.getKey();
                if(entry.getValue()==1){
                    it.remove();
                }else {
                    entry.setValue(entry.getValue()-1);
                }
                if(j!=1&&a[i]-a[i-1]!=1){
                    return false;
                }
                i++;
            }
        }
        return true;
    }
}
