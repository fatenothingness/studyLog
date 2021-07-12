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
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.stream.Collectors;

public class Test {

    public static void main(String args[]) {

        ThreadLocal<String> threadLocal = new ThreadLocal<>();

        MyContainer<String> myContainer =new MyContainer<>();
        //
        for(int i=0;i<10;i++){
            new Thread(()->{
                for(int j=0;j<10;j++){
                    System.out.println(myContainer.get());
                }
            },"cus"+i).start();
        }
        for(int i=0;i<2;i++){
            new Thread(()->{
                for(int j=0;j<50;j++){
                    myContainer.put(Thread.currentThread().getName()+":"+j);
                }
            },"pro"+i).start();
        }
    }

    static class MyContainer<T>{
        final private LinkedList<T> list =new LinkedList<>();
        final private Integer MAX = 10;
        private Integer count = 0;
        private Lock lock = new ReentrantLock();
        private Condition pro = lock.newCondition();
        private Condition cus = lock.newCondition();

        public void put(T t){
            try {
                lock.lock();
                while(list.size()==MAX){
                    try {
                        pro.await();
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
                list.add(t);
                ++count;
                cus.signalAll();
            }catch (Exception e){
                e.printStackTrace();
            }finally {
                lock.unlock();
            }
        }

        public T get(){
            T t =null;
            try {
                lock.lock();
                while (list.size()==0){
                    cus.await();
                }
                t = list.removeFirst();
                --count;
                pro.signalAll();
            }catch (Exception e){
                e.printStackTrace();
            }finally {
                lock.unlock();
            }
            return t;
        }
    }
}
