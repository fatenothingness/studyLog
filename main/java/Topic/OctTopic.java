package Topic;

import org.apache.poi.ss.formula.functions.T;

import java.util.*;

public class OctTopic {
    /**
     * 211. 添加与搜索单词 - 数据结构设计
     * 请你设计一个数据结构，支持 添加新单词 和 查找字符串是否与任何先前添加的字符串匹配 。
     *
     * 实现词典类 WordDictionary ：
     *
     * WordDictionary() 初始化词典对象
     * void addWord(word) 将 word 添加到数据结构中，之后可以对它进行匹配
     * bool search(word) 如果数据结构中存在字符串与 word 匹配，则返回 true ；否则，返回  false 。word 中可能包含一些 '.' ，每个 . 都可以表示任何一个字母。
     */
    class WordDictionary {

        class Node {
            Node[] tns = new Node[26];
            boolean isWord;
        }
        Node root;
        public WordDictionary() {
            root = new Node();
        }

        public void addWord(String word) {
            Node p = root;
            for(int i=0;i<word.length();i++){
                int c = word.charAt(i)-'a';
                if(p.tns[c]==null) p.tns[c] = new Node();
                p = p.tns[c];
            }
            p.isWord = true;
        }

        public boolean search(String word) {
            return dfs(word,root,0);
        }

        boolean dfs(String s,Node root,int idx){
            int n = s.length();
            if(idx==n) return root.isWord;
            char c = s.charAt(idx);
            if(c=='.'){
                for(int i=0;i<26;i++){
                    if(root.tns[i]!=null&&dfs(s,root.tns[i],idx+1)) return true;
                }
                return false;
            }else {
                int u = c-'a';
                if(root.tns[u]==null) return false;
                return dfs(s,root.tns[u],idx+1);
            }
        }
    }

    /**
     * 453. 最小操作次数使数组元素相等
     * 给你一个长度为 n 的整数数组，每次操作将会使 n - 1 个元素增加 1 。返回让数组所有元素相等的最小操作次数。
     * @param nums
     * @return
     */
    public int minMoves(int[] nums) {
        Arrays.sort(nums);
        int ans = 0;
        int n = nums.length;
        for (int i = 1; i < n; i++) {
            ans+=nums[i]-nums[0];
        }
        return ans;
    }


    /**
     * 66. 加一
     * 给定一个由 整数 组成的 非空 数组所表示的非负整数，在该数的基础上加一。
     *
     * 最高位数字存放在数组的首位， 数组中每个元素只存储单个数字。
     *
     * 你可以假设除了整数 0 之外，这个整数不会以零开头。
     */
    public int[] plusOne(int[] digits) {
        int len = digits.length;
        for(int i=len-1;i>=0;i--){
            if(digits[i]==9){
                digits[i]=0;
            }else {
                digits[i]=digits[i]+1;
                break;
            }
        }
        if(digits[0]==0){
            int[] n = new int[len+1];
            Arrays.fill(n,0);
            n[0]=1;
            return n;
        }else {
            return digits;
        }
    }

    /**
     * 496. 下一个更大元素 I
     * 给你两个 没有重复元素 的数组 nums1 和 nums2 ，其中nums1 是 nums2 的子集。
     *
     * 请你找出 nums1 中每个元素在 nums2 中的下一个比其大的值。
     *
     * nums1 中数字 x 的下一个更大元素是指 x 在 nums2 中对应位置的右边的第一个比 x 大的元素。如果不存在，对应位置输出 -1 。
     */
    public int[] nextGreaterElement(int[] nums1, int[] nums2) {
        Map<Integer,Integer> map = new HashMap<>();
        for(int i=0;i<nums2.length;i++){
            map.put(nums2[i],-1);
            for(int j = i+1;j<nums2.length;j++){
                if(nums2[j]>nums2[i]){
                    map.put(nums2[i],nums2[j]);
                    break;
                }
            }
        }
        int[] res = new int[nums1.length];
        for(int i=0;i<nums1.length;i++){
            res[i] = map.get(nums1[i]);
        }
        return res;
    }


    /**
     * 335. 路径交叉
     * 给你一个整数数组 distance 。
     *
     * 从 X-Y 平面上的点 (0,0) 开始，先向北移动 distance[0] 米，然后向西移动 distance[1] 米，向南移动 distance[2] 米，向东移动 distance[3] 米，持续移动。也就是说，每次移动后你的方位会发生逆时针变化。
     *
     * 判断你所经过的路径是否相交。如果相交，返回 true ；否则，返回 false 。
     */
    public boolean isSelfCrossing(int[] d) {
        int n = d.length;
        if (n < 4) return false;
        for (int i = 3; i < n; i++) {
            if (d[i] >= d[i - 2] && d[i - 1] <= d[i - 3]) return true;
            if (i >= 4 && d[i - 1] == d[i - 3] && d[i] + d[i - 4] >= d[i - 2]) return true;
            if (i >= 5 && d[i - 1] <= d[i - 3] && d[i - 2] > d[i - 4] && d[i] + d[i - 4] >= d[i - 2] && d[i - 1] + d[i - 5] >= d[i - 3]) return true;
        }
        return false;
    }

    /**
     * 384. 打乱数组
     * 给你一个整数数组 nums ，设计算法来打乱一个没有重复元素的数组。
     *
     * 实现 Solution class:
     *
     * Solution(int[] nums) 使用整数数组 nums 初始化对象
     * int[] reset() 重设数组到它的初始状态并返回
     * int[] shuffle() 返回数组随机打乱后的结果
     */
    public static class Solution {
        int [] init;
        int [] now;
        Random r = new Random(1);
        public Solution(int[] nums) {
            now = nums;
            init = nums.clone();
        }

        public int[] reset() {
            int len = now.length;
            int l = r.nextInt(len);
            int s = r.nextInt(len);
            int tmp = now[l];
            now[l] = now[s];
            now[s] = tmp;
            return now;
        }

        public int[] shuffle() {
            return init;
        }
    }

    public static void main(String args[]){
        Solution s = new Solution(new int[]{1,2,3,5,8,4});
        System.out.println(s.shuffle());
        System.out.println(s.reset());
        System.out.println(s.shuffle());
    }
}
