import { Component } from '@angular/core';
import { AddRepositoryComponent } from '../add-repository/add-repository.component';
import { Router } from '@angular/router';
import { BaseTemplateComponent } from '../base-template/base-template.component';

@Component({
  selector: 'app-home',
  standalone: true,
  imports: [BaseTemplateComponent, AddRepositoryComponent],
  templateUrl: './home.component.html',
  styleUrl: './home.component.css',
})
export class HomeComponent {

  constructor(private router: Router) { }

  goToReadyModels() {
    this.router.navigate(['/available-models']);
  }

}
