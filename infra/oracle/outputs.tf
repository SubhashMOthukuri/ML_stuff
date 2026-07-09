output "public_ip" {
  value = oci_core_instance.app.public_ip
}

output "ssh_command" {
  value = "ssh -i ~/.ssh/id_rsa opc@${oci_core_instance.app.public_ip}"
}
