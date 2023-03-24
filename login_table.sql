create database frs;
use  frs;
create table if not exists login_table(
username varchar(40) not null primary key,
password varchar(40) not null
);
insert into login_table(username,password)
values("pavanyendluri","Pavan@99499"),
("prudhviyendluri","Prudhvi@99499"),
("admin","1234");

