# spark ui: spark://ubuntu:7077

# start standalone master server
start-master.sh

# listening on port 8080
sudo ss -tunelp | grep 8080

# start slave worker process
start-slave.sh spark://ubuntu:7077

# if script not in PATH, use this to locate it and run with full path
sudo updatedb
locate start-slave.sh

# start spark shell
/opt/spark/bin/spark-shell

# OR start pyspark shell
/opt/spark/bin/pyspark

# shut down processes
SPARK_HOME/sbin/stop-slave.sh
SPARK_HOME/sbin/stop-master.sh