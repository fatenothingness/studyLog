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
        Student s = new Student();
        s.setName("zhangsan");
        ListNode a = new ListNode(1,new ListNode(2,new ListNode(4,null)));
        ListNode b = new ListNode(1,new ListNode(3,new ListNode(4,null)));
        SwordToOffer swordToOffer = new SwordToOffer();
        ListNode listNode = swordToOffer.mergeTwoLists(a, b);
        System.out.println(listNode);
    }
}
