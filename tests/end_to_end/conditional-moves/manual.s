glabel test
li      $a1,1
slt     $a0,$zero,$a2
movz    $a2,$a1,$a0
li      $v0,5
slti    $v1,$a2,6
jr      $ra
ext     $t0, $a0, 0, 16      
ins     $t1, $a1, 16, 16
seb     $t2, $a2
seh     $t3, $v0
movn    $v0,$a2,$v1
