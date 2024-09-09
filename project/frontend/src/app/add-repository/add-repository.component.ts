import { Component, OnInit } from '@angular/core';
import { FormControl, FormControlName, ReactiveFormsModule, FormGroup, Validators, FormBuilder } from '@angular/forms';
import { API_ROUTES } from '../../../api-routes';

@Component({
  selector: 'app-add-repository',
  standalone: true,
  imports: [ReactiveFormsModule],
  templateUrl: './add-repository.component.html',
  styleUrl: './add-repository.component.css'
})

export class AddRepositoryComponent implements OnInit{
  addRepositoryForm: FormGroup;

  constructor(private fb: FormBuilder) { 
    this.addRepositoryForm = this.fb.group({
      repository: ['', Validators.required],
      branch: ['', Validators.required]
    });
  }

  ngOnInit(): void { }

  async onSubmit(): Promise<void> {
    console.log(this.addRepositoryForm.value);

    const formData = new FormData();
    const repository = this.addRepositoryForm.value.repository;
    const branch = this.addRepositoryForm.value.branch;

    if (repository) {
      formData.append('repository_url', repository);
    }

    if (branch) {
      formData.append('branch', branch);
    }

    try{
      const response = await fetch(API_ROUTES.REPOSITORIES, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error('Error');
      }

      const result = await response.json();
      console.log(result);
    } catch (error) {
      console.error(error);
    }
  }
}
