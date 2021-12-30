import com.alibaba.fastjson.JSONObject;
import handle.*;
import javassist.util.proxy.ProxyFactory;
import net.sf.cglib.proxy.Enhancer;
import org.checkerframework.checker.units.qual.A;
import org.springframework.context.ApplicationContext;
import org.springframework.context.support.ClassPathXmlApplicationContext;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Proxy;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.stream.Collectors;

public class Test {
    static abstract class Human{
        public void sayHello(){
            System.out.println("hello human");
        }
    }
    static class Man extends Human{

        @Override
        public void sayHello(){
            System.out.println("hello man");
        }
    }
    static class Woman extends Human{
        @Override
        public void sayHello(){
            System.out.println("hello woman");
        }
    }



    public static void main(String args[]) {
        Human man = new Man();
        Woman woman = new Woman();
        Test test = new Test();
        man.sayHello();
        woman.sayHello();
    }
}
