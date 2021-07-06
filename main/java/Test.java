import com.alibaba.fastjson.JSONObject;
import handle.Person;
import handle.Student;
import handle.TestHandle;
import javassist.util.proxy.ProxyFactory;
import org.checkerframework.checker.units.qual.A;
import org.springframework.context.ApplicationContext;
import org.springframework.context.support.ClassPathXmlApplicationContext;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Proxy;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.stream.Collectors;

public class Test {

    volatile int a = 0;

    public static void main(String args[]) throws ClassNotFoundException, IllegalAccessException, InstantiationException {
        OneTopicEveryday oneTopicEveryday = new OneTopicEveryday();
        int[][] two = new int[][]{{1,2,7},{3,6,7}};
       // System.out.println(oneTopicEveryday.convertToTitle(701));
        SwordToOffer s= new SwordToOffer();
        TreeNode root = new TreeNode(0);
        TreeNode a = new TreeNode(2);
        TreeNode b = new TreeNode(4);
        TreeNode c = new TreeNode(1);
        TreeNode d = new TreeNode(3);
        TreeNode e = new TreeNode(-1);
        TreeNode f = new TreeNode(5);
        TreeNode g = new TreeNode(1);
        TreeNode h = new TreeNode(6);
        TreeNode m = new TreeNode(8);
        root.left=a;root.right=b;
        a.left=c;
        b.left =d; b.right=e;
        c.left=f; c.right=g;
        d.right=h;
        e.right = m;

        int[] t =new int[]{1,2,3,2,2,2,5,4,2};
        System.out.println(s.firstUniqChar("abaccdeff"));
    }
}
