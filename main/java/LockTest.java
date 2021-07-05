import java.util.concurrent.CountDownLatch;

public class LockTest{


    public static void main(String args[]){
         CountDownLatch countDownLatch = new CountDownLatch(1);

         Thread t1 = new Thread(new Runnable() {
             @Override
             public void run() {
                 for(int i=0;i<10;i++){
                     System.out.println(Thread.currentThread().getName()+"第"+i);
                     if(i==5){
                         try{
                             countDownLatch.await();
                         }catch (Exception e){
                             e.printStackTrace();
                         }
                     }
                 }
             }
         });

        Thread t2 = new Thread(new Runnable() {
            @Override
            public void run() {
                for(int i=0;i<100;i++){
                    System.out.println(Thread.currentThread().getName()+"第"+i);
                }
                try {
                    Thread.sleep(1000);
                }catch (Exception e){
                    e.printStackTrace();
                }
                countDownLatch.countDown();
            }
        });
        t1.start();
        try {
            Thread.sleep(1000);
        }catch (Exception e){
            e.printStackTrace();
        }
        t2.start();
    }
}
