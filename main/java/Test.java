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
import java.util.stream.Collectors;

public class Test {


    public static void main(String args[]){
        OneTopicEveryday oneTopicEveryday = new OneTopicEveryday();
        int[][] two = new int[][]{{1,2,7},{3,6,7}};
        int i = oneTopicEveryday.numBusesToDestination(two, 1, 6);
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
        System.out.println(s.levelOrder3(root));

    }
}
