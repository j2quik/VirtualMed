<?php

$username = $_POST['username'];
$password = $_POST['password'];
$firstname = $_POST['firstName'];
$lastname = $_POST['lastname'];
$phone_number = $_POST['phone_number'];
$email = $_POST['email'];

//Database Connection
$conn = new mysqli('localhost', 'root','','test');
if($conn->connect_error){

    die('connection Failed : '. $conn->connect_error);
}else{
    $stmt = $conn->prepare("insert into registration(username,password,firstName,lastname,phone_number,email)
    values (?, ?, ?, ?, ?, ?,)")
    $stmt->bind_param("ssssis", $username,$password,$firstname,$lastname,$phone_number,$email);
    $stmt->execute();
    echo "registration Successfully";
    $stmt->close();
    $conn->close();
}

?>
