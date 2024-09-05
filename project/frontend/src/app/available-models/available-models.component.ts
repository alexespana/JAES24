import { Component, OnInit } from '@angular/core';
import { BaseTemplateComponent } from '../base-template/base-template.component';
import { RepositoriesService } from '../repositories.service';
import { CommonModule } from '@angular/common';

interface Repository {
  repository: string;
  branch: string;
  features_file: string;
  pickle_pattern: string;
  available: string;
}

@Component({
  selector: 'app-available-models',
  standalone: true,
  imports: [BaseTemplateComponent, CommonModule],
  templateUrl: './available-models.component.html',
  styleUrl: './available-models.component.css'
})
export class AvailableModelsComponent implements OnInit {
  repositories: Repository[] = [];
  pagination = {
    page: 1,
    pageSize: 5,
    totalItems: 0
  };

  constructor(private repositoryService: RepositoriesService) { }

  ngOnInit(): void {
      this.repositoryService.getAvailableModels().subscribe(
        (data: Repository[]) => {
          this.repositories = data;
          this.pagination.totalItems = data.length;
        },
        (error) => {
          console.error('Error al obtener los repositorios', error);
        }
      );
  }

  get pagedRepositories(): Repository[] {
    const start = (this.pagination.page - 1) * this.pagination.pageSize;
    return this.repositories.slice(start, start + this.pagination.pageSize);
  }

  setPage(page: number): void {
    if (page > 0 && page <= this.totalPages) {
      this.pagination.page = page;
    }
  }

  get totalPages(): number {
    return Math.ceil(this.pagination.totalItems / this.pagination.pageSize);
  }
}
