package Topic;

import util.ListNode;

import java.util.*;

public class SeptemberTopic {
    /**
     * 面试题 17.14. 最小K个数
     * 设计一个算法，找出数组中最小的k个数。以任意顺序返回这k个数均可。
     * @param arr
     * @param k
     * @return
     */
    int k;
    public int[] smallestK(int[] arr, int _k) {
        k = _k;
        int n = arr.length;
        int[] ans = new int[k];
        if (k == 0) return ans;
        qsort(arr, 0, n - 1);
        for (int i = 0; i < k; i++) ans[i] = arr[i];
        return ans;
    }
    void qsort(int[] arr, int l, int r) {
        if (l >= r) return ;
        int i = l, j = r;
        int ridx = new Random().nextInt(r - l + 1) + l;
        swap(arr, ridx, l);
        int x = arr[l];
        while (i < j) {
            while (i < j && arr[j] >= x) j--;
            while (i < j && arr[i] <= x) i++;
            swap(arr, i, j);
        }
        swap(arr, i, l);
        // 集中答疑：因为题解是使用「基准点左侧」来进行描述（不包含基准点的意思），所以这里用的 k（写成 k - 1 也可以滴
        if (i > k) qsort(arr, l, i - 1);
        if (i < k) qsort(arr, i + 1, r);
    }
    void swap(int[] arr, int l, int r) {
        int tmp = arr[l];
        arr[l] = arr[r];
        arr[r] = tmp;
    }


    public int search(int[] nums, int target) {
        if(nums.length==1){
            return nums[0]==target?0:-1;
        }
        int l = 0;
        int r = nums.length-1;
        while(l<r){
            int mid = (l+r)/2;
            if(target>nums[mid]){
                l = mid+1;
            }else if(target<nums[mid]){
                r = mid;
            }else{
                return mid;
            }
        }
        return -1;
    }


    /**
     * 1221. 分割平衡字符串
     * 在一个 平衡字符串 中，'L' 和 'R' 字符的数量是相同的。
     *
     * 给你一个平衡字符串 s，请你将它分割成尽可能多的平衡字符串。
     *
     * 注意：分割得到的每个字符串都必须是平衡字符串。
     *
     * 返回可以通过分割得到的平衡字符串的 最大数量 。
     */
    public int balancedStringSplit(String s) {
        int res = 0;
        int t = 0;
        for(int i=0;i<s.length();i++){
            char a = s.charAt(i);
            if(a=='L'){
                t++;
            }else {
                t--;
            }
            if(t==0){
                res++;
            }
        }
        return res;
    }

    /**
     * 502. IPO
     * 假设 力扣（LeetCode）即将开始 IPO 。为了以更高的价格将股票卖给风险投资公司，力扣 希望在 IPO 之前开展一些项目以增加其资本。 由于资源有限，它只能在 IPO 之前完成最多 k 个不同的项目。
     * 帮助 力扣 设计完成最多 k 个不同项目后得到最大总资本的方式。
     *
     * 给你 n 个项目。对于每个项目 i ，它都有一个纯利润 profits[i] ，和启动该项目需要的最小资本 capital[i] 。
     *
     * 最初，你的资本为 w 。当你完成一个项目时，你将获得纯利润，且利润将被添加到你的总资本中。
     *
     * 总而言之，从给定项目中选择 最多 k 个不同项目的列表，以 最大化最终资本 ，并输出最终可获得的最多资本。
     *
     * 答案保证在 32 位有符号整数范围内。
     */
    class Project{
        public int id;
        public int p;
        public int c;
        public Project(int id,int p,int c){
            this.id = id;
            this.p = p;
            this.c = c;
        }
        public Project(){

        }
    }
    public int findMaximizedCapital(int k, int w, int[] profits, int[] capital) {
        Queue<Project> queue1 = new PriorityQueue<>((o1, o2) -> o1.c-o2.c);
        for(int i=0;i<profits.length;i++){
            Project p = new Project(i,profits[i],capital[i]);
            queue1.add(p);
        }
        Queue<Project> queue = new PriorityQueue<>((o1, o2) -> o2.p-o1.p);
        while(k>0){
            k--;
            while(!queue1.isEmpty()){
                if(queue1.peek().c<=w){
                    queue.add(queue1.poll());
                }else {
                    break;
                }
            }
            if(!queue.isEmpty()){
                w+=queue.poll().p;
            }else {
                break;
            }
        }
        return w;
    }

    /**
     * 68. 文本左右对齐
     * 给定一个单词数组和一个长度 maxWidth，重新排版单词，使其成为每行恰好有 maxWidth 个字符，且左右两端对齐的文本。
     *
     * 你应该使用“贪心算法”来放置给定的单词；也就是说，尽可能多地往每行中放置单词。必要时可用空格 ' ' 填充，使得每行恰好有 maxWidth 个字符。
     *
     * 要求尽可能均匀分配单词间的空格数量。如果某一行单词间的空格不能均匀分配，则左侧放置的空格数要多于右侧的空格数。
     *
     * 文本的最后一行应为左对齐，且单词之间不插入额外的空格。
     */
    public List<String> fullJustify(String[] words, int maxWidth) {
        List<String> res = new ArrayList<>();
        Queue<String> queue = new LinkedList<>();
        int length = 0;
        int word = 0;
        int i = 0;
        while(i<words.length){
            String s = words[i];
           if(length+s.length()<=maxWidth){
               queue.add(s);
               length += s.length()+1;
               word += s.length();
               i++;
           }else {
               StringBuilder sb = new StringBuilder();
               int size = queue.size();
               //计算每个单词之间的空格有多少个
               int sum;
               int m;
               if(size==1){
                   sum =maxWidth-word;
                   m = 0;
               }else {
                   sum =(maxWidth-word)/(size-1);
                   m = (maxWidth-word)%(size-1);
               }
               while(!queue.isEmpty()){
                   sb.append(queue.poll());
                   if(size==1){
                       for(int j=0;j<sum;j++) sb.append(" ");
                   }
                   if(!queue.isEmpty()){
                       for(int j=0;j<sum;j++) sb.append(" ");
                       if(m-->0) sb.append(" ");
                   }
               }
               res.add(sb.toString());
               length=0;
               word =0;
           }
        }
        //处理最后一行的数据
        if(!queue.isEmpty()){
            StringBuilder sb = new StringBuilder();
            while(!queue.isEmpty()){
                sb.append(queue.poll());
                if(!queue.isEmpty()){
                    sb.append(" ");
                }
            }
            for(int j=sb.length();j<maxWidth;j++) sb.append(" ");
            res.add(sb.toString());
        }
        return res;
    }

    /**
     * 162. 寻找峰值
     * 峰值元素是指其值严格大于左右相邻值的元素。
     *
     * 给你一个整数数组 nums，找到峰值元素并返回其索引。数组可能包含多个峰值，在这种情况下，返回 任何一个峰值 所在位置即可。
     *
     * 你可以假设 nums[-1] = nums[n] = -∞ 。
     *
     * 你必须实现时间复杂度为 O(log n) 的算法来解决此问题。
     */
    public int findPeakElement(int[] nums) {
        int n = nums.length;
        int l = 0, r = n - 1;
        while (l < r) {
            int mid = l + r >> 1;
            if (nums[mid] > nums[mid + 1]) r = mid;
            else l = mid + 1;
        }
        return r;
    }

    /**
     * 725. 分隔链表
     * 给你一个头结点为 head 的单链表和一个整数 k ，请你设计一个算法将链表分隔为 k 个连续的部分。
     *
     * 每部分的长度应该尽可能的相等：任意两部分的长度差距不能超过 1 。这可能会导致有些部分为 null 。
     *
     * 这 k 个部分应该按照在链表中出现的顺序排列，并且排在前面的部分的长度应该大于或等于排在后面的长度。
     *
     * 返回一个由上述 k 部分组成的数组。
     */
    public ListNode[] splitListToParts(ListNode head, int k) {
        ListNode[] listNodes = new ListNode[k];
        ListNode root = head;
        int size = 0;
        while(root!=null){
            size++;
            root = root.next;
        }
        int len = size/k;
        int t = size%k;
        int i=0;
        ListNode l1 = head;
        while(i<k){
            ListNode f = l1;
            ListNode l2 = new ListNode(1,l1);
            int j=0;
            while(j<len){
                l1=l1.next;
                l2=l2.next;
                j++;
            }
            if(t>0){
                l1=l1.next;
                l2=l2.next;
                t--;
            }
            l2.next=null;
            listNodes[i++] = f;
        }
        return listNodes;
    }

    /**
     * 29. 两数相除
     * 给定两个整数，被除数 dividend 和除数 divisor。将两数相除，要求不使用乘法、除法和 mod 运算符。
     *
     * 返回被除数 dividend 除以除数 divisor 得到的商。
     *
     * 整数除法的结果应当截去（truncate）其小数部分，例如：truncate(8.345) = 8 以及 truncate(-2.7335) = -2
     */
    int INF = Integer.MAX_VALUE;
    public int divide(int _a, int _b) {
        long a = _a, b = _b;
        boolean flag = false;
        if ((a < 0 && b > 0) || (a > 0 && b < 0)) flag = true;
        if (a < 0) a = -a;
        if (b < 0) b = -b;
        long l = 0, r = a;
        while (l < r) {
            long mid = l + r + 1 >> 1;
            if (mul(mid, b) <= a) l = mid;
            else r = mid - 1;
        }
        r = flag ? -r : r;
        if (r > INF || r < -INF - 1) return INF;
        return (int)r;
    }
    long mul(long a, long k) {
        long ans = 0;
        while (k > 0) {
            if ((k & 1) == 1) ans += a;
            k >>= 1;
            a += a;
        }
        return ans;
    }
}
