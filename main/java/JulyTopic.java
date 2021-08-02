import java.util.*;

public class JulyTopic {
    public static final int mod = (int)1e9+7;

    /**
     * 1818. 绝对差值和
     * 给你两个正整数数组 nums1 和 nums2 ，数组的长度都是 n 。
     *
     * 数组 nums1 和 nums2 的 绝对差值和 定义为所有 |nums1[i] - nums2[i]|（0 <= i < n）的 总和（下标从 0 开始）。
     *
     * 你可以选用 nums1 中的 任意一个 元素来替换 nums1 中的 至多 一个元素，以 最小化 绝对差值和。
     *
     * 在替换数组 nums1 中最多一个元素 之后 ，返回最小绝对差值和。因为答案可能很大，所以需要对 109 + 7 取余 后返回。
     */
    public int minAbsoluteSumDiff(int[] nums1, int[] nums2) {
        int n = nums1.length;
        int[] sorted = nums1.clone();
        Arrays.sort(sorted);
        long sum = 0, max = 0;
        for (int i = 0; i < n; i++) {
            int a = nums1[i], b = nums2[i];
            if (a == b) continue;
            int x = Math.abs(a - b);
            sum += x;
            int l = 0, r = n - 1;
            while (l < r) {
                int mid = l + r + 1 >> 1;
                if (sorted[mid] <= b) l = mid;
                else r = mid - 1;
            }
            int nd = Math.abs(sorted[r] - b);
            if (r + 1 < n) nd = Math.min(nd, Math.abs(sorted[r + 1] - b));
            if (nd < x) max = Math.max(max, x - nd);
        }
        return (int)((sum - max) % mod);
    }

    /**
     * 1846. 减小和重新排列数组后的最大元素
     * 给你一个正整数数组 arr 。请你对 arr 执行一些操作（也可以不进行任何操作），使得数组满足以下条件：
     * arr 中 第一个 元素必须为 1 。
     * 任意相邻两个元素的差的绝对值 小于等于 1 ，也就是说，对于任意的 1 <= i < arr.length （数组下标从 0 开始），都满足 abs(arr[i] - arr[i - 1]) <= 1 。abs(x) 为 x 的绝对值。
     */
    public int maximumElementAfterDecrementingAndRearranging(int[] arr) {
        Arrays.sort(arr);
        int n = arr.length;
        arr[0] = 1;
        for(int i=1;i<n;i++){
            if(arr[i]-arr[i-1]>1){
                arr[i]=arr[i-1]+1;
            }
        }
        return arr[n-1];
    }


    /**
     * 剑指 Offer 53 - I. 在排序数组中查找数字 I
     * 统计一个数字在排序数组中出现的次数。
     */
    public int search(int[] nums, int target) {
        int l = 0;
        int r = nums.length-1;
        while(l<r){
            int mid = l+(r-l>>1);
            if(nums[mid]>=target){
                r=mid;
            }else {
                l = mid+1;
            }
        }
        int res = 0;
        for(int i=l;i<nums.length;i++){
            if(nums[i]==target){
                res++;
            }else {
                break;
            }
        }
        return res;
    }

    /**
     * 1838. 最高频元素的频数
     * 元素的 频数 是该元素在一个数组中出现的次数。
     *
     * 给你一个整数数组 nums 和一个整数 k 。在一步操作中，你可以选择 nums 的一个下标，并将该下标对应元素的值增加 1 。
     *
     * 执行最多 k 次操作后，返回数组中最高频元素的 最大可能频数 。
     *
     *
     */
    public int maxFrequency(int[] nums, int k) {
        int n = nums.length;
        Arrays.sort(nums);
        int res = 1;
        int left = n-2;
        int right  = n-1;
        while(left>=0){
            if(nums[right]==nums[left]||nums[right]-nums[left]<=k){
                res = Math.max(res,right-left+1);
                k-=nums[right]-nums[left];
                left--;
            }else {
                k+=(nums[right]-nums[right-1])*(right-left-1);
                right--;
            }
        }
        return res;
    }

    /**
     * 1877. 数组中最大数对和的最小值
     * 一个数对 (a,b) 的 数对和 等于 a + b 。最大数对和 是一个数对数组中最大的 数对和 。
     *
     * 比方说，如果我们有数对 (1,5) ，(2,3) 和 (4,4)，最大数对和 为 max(1+5, 2+3, 4+4) = max(6, 5, 8) = 8 。
     * 给你一个长度为 偶数 n 的数组 nums ，请你将 nums 中的元素分成 n / 2 个数对，使得：
     *
     * nums 中每个元素 恰好 在 一个 数对中，且
     * 最大数对和 的值 最小 。
     * 请你在最优数对划分的方案下，返回最小的 最大数对和 。
     */

    public int minPairSum(int[] nums) {
        int n = nums.length;
        Arrays.sort(nums);
        int res = 0;
        for(int i=0;i<n;i++){
            res = Math.max(nums[i]+nums[n-i-1],res);
        }
        return res;
    }

    class Node {
        int val;
        Node next;
        Node random;

        public Node(int val) {
            this.val = val;
            this.next = null;
            this.random = null;
        }
    }

    /**
     * 138. 复制带随机指针的链表
     * 给你一个长度为 n 的链表，每个节点包含一个额外增加的随机指针 random ，该指针可以指向链表中的任何节点或空节点。
     * 构造这个链表的 深拷贝。 深拷贝应该正好由 n 个 全新 节点组成，其中每个新节点的值都设为其对应的原节点的值。新节点的 next 指针和 random 指针也都应指向复制链表中的新节点，并使原链表和复制链表中的这些指针能够表示相同的链表状态。复制链表中的指针都不应指向原链表中的节点 。
     */
    public Node copyRandomList(Node head) {
        if(head==null){
            return null;
        }
        HashMap<Node,Integer> map =new HashMap<>();
        List<Node> list =new ArrayList<>();
        Node t = head;
        int i=0;
        while(t!=null){
            map.put(t,i++);
            Node node = new Node(t.val);
            list.add(node);
            t=t.next;
        }
        for(int j=0;j<list.size()-1;j++){
            Node now = list.get(j);
            Node next = list.get(j+1);
            now.next=next;
        }
        int j=0;
        while(head!=null){
            if(head.random==null){
                list.get(j++).random=null;
            }else {
                Node random = list.get(map.get(head.random));
                list.get(j++).random=random;
            }
            head=head.next;
        }
        return list.get(0);
    }

    /**
     * 1893. 检查是否区域内所有整数都被覆盖
     * 给你一个二维整数数组 ranges 和两个整数 left 和 right 。每个 ranges[i] = [starti, endi] 表示一个从 starti 到 endi 的 闭区间 。
     *
     * 如果闭区间 [left, right] 内每个整数都被 ranges 中 至少一个 区间覆盖，那么请你返回 true ，否则返回 false 。
     *
     * 已知区间 ranges[i] = [starti, endi] ，如果整数 x 满足 starti <= x <= endi ，那么我们称整数x 被覆盖了。
     */
    public boolean isCovered(int[][] ranges, int left, int right) {
        int[] num = new int[right+1];
        for(int[] a:ranges){
            for(int i=a[0];i<=a[1];i++){
               if(i<=right){
                   num[i]=1;
               }else {
                   break;
               }
            }
        }
        for(int i=left;i<=right;i++){
            if(num[i]!=1){
                return false;
            }
        }
        return true;
    }

    /**
     * 671. 二叉树中第二小的节点
     * 给定一个非空特殊的二叉树，每个节点都是正数，并且每个节点的子节点数量只能为 2 或 0。如果一个节点有两个子节点的话，那么该节点的值等于两个子节点中较小的一个。
     *
     * 更正式地说，root.val = min(root.left.val, root.right.val) 总成立。
     *
     * 给出这样的一个二叉树，你需要输出所有节点中的第二小的值。如果第二小的值不存在的话，输出 -1 。
     */
    int f ;
    int ans ;
    public int findSecondMinimumValue(TreeNode root) {
        if (root==null){
            return -1;
        }
        f = root.val;
        ans = -1;
        process(root);
        return ans;
    }
    private void process(TreeNode root){
        if(root==null){
            return;
        }
        if(ans!=-1&&root.val>=ans){
            return;
        }
        if(root.val>f){
            ans=root.val;
        }
        process(root.left);
        process(root.right);
    }

    /**
     * 863. 二叉树中所有距离为 K 的结点
     * 给定一个二叉树（具有根结点 root）， 一个目标结点 target ，和一个整数值 K 。
     *
     * 返回到目标结点 target 距离为 K 的所有结点的值的列表。 答案可以以任何顺序返回。
     */
    public List<Integer> distanceK(TreeNode root, TreeNode target, int k) {
        Map<TreeNode,TreeNode> map = new HashMap<>();
        Set<TreeNode> set =new HashSet<>();
        //dps，用哈希表保存每个节点的父节点信息
        dps863(root,map);
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(target);
        set.add(target);
        List<Integer> res = new ArrayList<>();
        //广度优先遍历，从target出发，向左右父节点寻找，用set记录已经路过的节点。
        while(k>0){
            k--;
            int size = queue.size();
            while(size>0){
                TreeNode now = queue.poll();
                TreeNode last = map.get(now);
                if(now.left!=null&&!set.contains(now.left)){
                    queue.add(now.left);
                    set.add(now.left);
                }
                if(now.right!=null&&!set.contains(now.right)){
                    queue.add(now.right);
                    set.add(now.right);
                }
                if(last!=null&&!set.contains(last)){
                    queue.add(last);
                    set.add(last);
                }
                size--;
            }
        }
        while(!queue.isEmpty()){
            res.add(queue.poll().val);
        }
        return res;
    }

    private void dps863(TreeNode root,Map<TreeNode,TreeNode> map){
        if(root==null){
            return;
        }
        if(root.left!=null){
            map.put(root.left,root);
        }
        if(root.right!=null){
            map.put(root.right,root);
        }
        dps863(root.left,map);
        dps863(root.right,map);
    }

    /**
     * 1104. 二叉树寻路
     * 在一棵无限的二叉树上，每个节点都有两个子节点，树中的节点 逐行 依次按 “之” 字形进行标记。
     *
     * 如下图所示，在奇数行（即，第一行、第三行、第五行……）中，按从左到右的顺序进行标记；
     *
     * 而偶数行（即，第二行、第四行、第六行……）中，按从右到左的顺序进行标记。
     */
    public List<Integer> pathInZigZagTree(int label) {
        int row = 1, rowStart = 1;
        while (rowStart * 2 <= label) {
            row++;
            rowStart *= 2;
        }
        if (row % 2 == 0) {
            label = getReverse(label, row);
        }
        List<Integer> path = new ArrayList<Integer>();
        while (row > 0) {
            if (row % 2 == 0) {
                path.add(getReverse(label, row));
            } else {
                path.add(label);
            }
            row--;
            label >>= 1;
        }
        Collections.reverse(path);
        return path;
    }

    public int getReverse(int label, int row) {
        return (1 << row - 1) + (1 << row) - 1 - label;
    }

    /**
     * 171. Excel表列序号
     * 给你一个字符串 columnTitle ，表示 Excel 表格中的列名称。返回该列名称对应的列序号。
     */
    public int titleToNumber(String columnTitle) {
        int res = 0;
        int size = columnTitle.length();
        for(int i=0;i<size;i++){
            char n = columnTitle.charAt(i);
            int a = n-'A'+1;
            res = res*26+a;
        }
        return res;
    }
}
