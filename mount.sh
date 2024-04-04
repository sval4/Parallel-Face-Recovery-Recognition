#!/usr/bin/expect

set password "Sv201789@"

send "sudo umount ~/mounts/home/scratch\r"

# Sometimes it does not ask for password

expect {
    "password for vals:" {
        send "Sv2019384756@22\r"
    }
    eof{

    }    
}

spawn sshfs cci03:/gpfs/u/home/PCPE/PCPEvlsh ~/mounts/home/scratch -o follow_symlinks

expect "(PCPEvlsh@blp03.ccni.rpi.edu) Password:"

send "$password\r"

expect "Passcode or option (1-1):"

send "1\r"

# send "cp highlife.cu ~/mounts/home/scratch\r"


sudo umount ~/mounts/home
Sv2019384756@22
sshfs cci01:/gpfs/u/scratch/PCPE/PCPEvlsh ~/mounts/home -o follow_symlinks
Sv201789@
cp face.c ~/mounts/home
cp face.cu ~/mounts/home
cp run.mk ~/mounts/home
cp faces_train360.csv ~/mounts/home
cp faces_test.csv ~/mounts/home
cp faces_train360x2.csv ~/mounts/home
cp faces_train360x4.csv ~/mounts/home
cp faces_train360x8.csv ~/mounts/home
cp faces_train360x16.csv ~/mounts/home

rm FACE*
split -b 3k face.c FACE
cp FACE* ~/mounts/home/scratch
cat FACE* > face.c